import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

import joblib
from learner import Learner
from modeling import ViterbiDecoder
from preprocessor import Corpus, EntityTypes
from utils import set_seed
from config import myArgs


def get_label_list(args):
    # prepare dataset
    if args.tagging_scheme == "BIOES":
        label_list = ["O", "B", "I", "E", "S"]
    elif args.tagging_scheme == "BIO":
        label_list = ["O", "B", "I"]
    else:
        label_list = ["O", "I"]
    return label_list


def get_data_path(args, train_mode: str):
    assert args.dataset in [
        "FewNERD",
        "Domain",
        "Domain2",
    ], f"Dataset: {args.dataset} Not Support."
    if args.dataset == "FewNERD":
        return os.path.join(
            args.data_path,
            args.mode,
            "{}_{}_{}.jsonl".format(train_mode, args.N, args.K),
        )
    elif args.dataset == "Domain":
        if train_mode == "dev":
            train_mode = "valid"
        text = "_shot_5" if args.K == 5 else ""
        replace_text = "-" if args.K == 5 else "_"
        return os.path.join(
            "dataset/ACL2020data",
            "xval_ner{}".format(text),
            "ner_{}_{}{}.json".format(train_mode, args.N, text).replace(
                "_", replace_text
            ),
        )
    elif args.dataset == "Domain2":
        if train_mode == "train":
            return os.path.join("domain2", "{}_10_5.json".format(train_mode))
        return os.path.join(
            "domain2", "{}_{}_{}.json".format(train_mode, args.mode, args.K)
        )


def replace_type_embedding(learner, args):
    logger.info("********** Replace trained type embedding **********")
    entity_types = joblib.load(os.path.join(args.result_dir, "type_embedding.pkl"))
    N, H = entity_types.types_embedding.weight.data.shape
    for ii in range(N):
        learner.models.embeddings.word_embeddings.weight.data[
            ii + 1
        ] = entity_types.types_embedding.weight.data[ii]


def train(args):
    logger.info("********** Scheme: Training **********")
    label_list = get_label_list(args)

    valid_data_path = get_data_path(args, "dev")
    valid_corpus = Corpus(
        logger,
        valid_data_path,
        args.bert_model,
        args.max_seq_len,
        label_list,
        args.entity_types,
        do_lower_case=True,
        shuffle=False,
        viterbi=args.viterbi,
        tagging=args.tagging_scheme,
        device=args.device,
        concat_types=args.concat_types,
        dataset=args.dataset,
    )

    if args.debug:
        test_corpus = valid_corpus
        train_corpus = valid_corpus
    else:
        train_data_path = get_data_path(args, "train")
        train_corpus = Corpus(
            logger,
            train_data_path,
            args.bert_model,
            args.max_seq_len,
            label_list,
            args.entity_types,
            do_lower_case=True,
            shuffle=True,
            tagging=args.tagging_scheme,
            device=args.device,
            concat_types=args.concat_types,
            dataset=args.dataset,
        )

        if (not args.ignore_eval_test) or args.load_best_and_test:
            test_data_path = get_data_path(args, "test")
            test_corpus = Corpus(
                logger,
                test_data_path,
                args.bert_model,
                args.max_seq_len,
                label_list,
                args.entity_types,
                do_lower_case=True,
                shuffle=False,
                viterbi=args.viterbi,
                tagging=args.tagging_scheme,
                device=args.device,
                concat_types=args.concat_types,
                dataset=args.dataset,
            )
    learner = Learner(
        args.bert_model,
        label_list,
        args.freeze_layer,
        logger,
        args.lr,
        args.warmup_prop,
        args.max_training_steps,
        py_alias=args.py_alias,
        args=args,
    )

    if "embedding" in args.concat_types:
        replace_type_embedding(learner, args)

    t = time.time()
    F1_valid_best = {ii: -1.0 for ii in ["mt"]}
    F1_test = -1.0
    
    protect_step = 400
    
    # only_train_spans = True
    for step in range(args.max_training_steps):
        # progress = 1.0 * step / args.max_training_steps

        batch_query, batch_support = train_corpus.get_batch_meta(
            batch_size=args.batch_size
        )
        
        # test(args, learner, test_corpus, "test", train_corpus)

        span_loss, type_loss, base_ce_loss, max_ce_loss, span_cl_loss, coarse_type_loss, fine_type_loss, type_cl_loss, fine_margin_loss = learner.forward(
            batch_query,
            batch_support
        )

        if (step+1) % 20 == 0:
            logger.info(
                "Step: {}/{}, span loss = {:.6f}, type loss = {:.6f}, base_ce_loss = {:.6f}, max_ce_loss = {:.6f}, span_cl_loss = {:.6f}, coarse type loss = {:.6f}, fine type loss = {:.6f}, type cl loss = {:.6f}, fine margin loss = {:.6f}, time = {:.2f}s.".format(
                    step+1, args.max_training_steps, span_loss, type_loss, base_ce_loss, max_ce_loss, span_cl_loss, coarse_type_loss, fine_type_loss, type_cl_loss, fine_margin_loss, time.time() - t
                )
            )
            
        # if (step + 1) % 100 == 0:
        #     torch.cuda.empty_cache()

        if (step+1) % args.eval_every_steps == 0 and step > protect_step:
            torch.cuda.empty_cache()
            logger.info("********** Scheme: evaluate - [valid] **********")
            result_valid, predictions_valid = test(args, learner, valid_corpus, "valid", train_corpus)

            F1_valid = result_valid["f1"]
                    
            if F1_valid > F1_valid_best['mt']:
                F1_valid_best['mt'] = F1_valid
                learner.save_model(args.result_dir, "en", args.max_seq_len, "mt")
                logger.info(f"Best multitask model Store {step + 1}")

    if args.load_best_and_test: # 结束加载best模型 test
        torch.cuda.empty_cache()
        test(args, learner, test_corpus, "test", load_best_and_test=True)

def test(args, learner, corpus, types: str, load_best_and_test=False):
    if corpus.viterbi != "none":
        id2label = corpus.id2label
        transition_matrix = corpus.transition_matrix
        if args.viterbi == "soft":
            label_list = get_label_list(args)
            train_data_path = get_data_path(args, "train")
            
            train_corpus = Corpus(
                logger,
                train_data_path,
                args.bert_model,
                args.max_seq_len,
                label_list,
                args.entity_types,
                do_lower_case=True,
                shuffle=True,
                tagging=args.tagging_scheme,
                viterbi="soft",
                device=args.device,
                concat_types=args.concat_types,
                dataset=args.dataset
            )
            id2label = train_corpus.id2label
            transition_matrix = train_corpus.transition_matrix

        viterbi_decoder = ViterbiDecoder(id2label, transition_matrix)
    else:
        viterbi_decoder = None
    
    result_test, predictions = learner.evaluate_mt(
        corpus,
        logger,
        lr=args.lr_finetune,
        ft_steps=args.max_ft_steps,
        mode=args.mode,
        set_type=types,
        viterbi_decoder=viterbi_decoder,
        load_best_and_test=load_best_and_test,
    )
    return result_test, predictions


def evaluate(args):
    logger.info("********** Scheme: Testing **********")
    label_list = get_label_list(args)
    test_data_path = get_data_path(args, "test")
        
    test_corpus = Corpus(
        logger,
        test_data_path,
        args.bert_model,
        args.max_seq_len,
        label_list,
        args.entity_types,
        do_lower_case=True,
        shuffle=False,
        tagging=args.tagging_scheme,
        viterbi=args.viterbi,
        concat_types=args.concat_types,
        dataset=args.dataset,
    )

    learner = Learner(
        args.bert_model,
        label_list,
        args.freeze_layer,
        logger,
        args.lr,
        args.warmup_prop,
        args.max_training_steps,
        model_dir=args.model_dir,
        py_alias=args.py_alias,
        args=args,
    )
    
    # logger.info("********** Scheme: evaluate - [valid] **********")
    # test(args, learner, valid_corpus, "valid")

    logger.info("********** Scheme: evaluate - [test] **********")
    test(args, learner, test_corpus, "test")


if __name__ == "__main__":

    args = myArgs()

    if "Domain" in args.dataset:
        args.types_path = "data/entity_types_domain.json"

    # setup random seed
    set_seed(args.seed, args.gpu_device)

    # set up GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    device = torch.device("cuda")
    
    # torch.cuda.set_device(args.gpu_device)
    
    if args.debug:
        os.makedirs("debug", exist_ok=True)
        top_dir = "debug/models-{}-{}-{}".format(args.N, args.K, args.mode)
    else:
        top_dir = "outputs/models-{}-{}-{}".format(args.N, args.K, args.mode)

    if args.mt_test_only: # multitask test
        args.model_dir = args.mt_model_dir
        
        if not os.path.exists(args.model_dir):
            if args.convert_bpe:
                os.makedirs(args.model_dir)
            else:
                raise ValueError("Model directory does not exist!")   
        
        test_file_name = f"test-ftlr_{args.ft_lr_span}" \
            + f"_{args.ft_lr_type}-ftsteps_{args.max_ft_steps}" \
            + f"_ftspan-step_{args.only_ft_span_steps}"
            
        store_test_dir = f"{args.model_dir}/{test_file_name}"
        if not os.path.exists(store_test_dir):
            os.makedirs(store_test_dir)
            
        fh = logging.FileHandler(f"{store_test_dir}/log-test.txt")
            
        with Path("{}/args-test.json".format(store_test_dir)).open("w", encoding="utf-8") as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
    
    else:
        args.result_dir = "bs_{}-lr_{}_{}-steps_{}-seed_{}{}".format(
            args.batch_size,
            args.lr_span,
            args.lr_type,
            args.max_training_steps,
            args.seed,
            "-{}".format(args.name) if args.name else "",
        )
        os.makedirs(top_dir, exist_ok=True)
        if not os.path.exists("{}/{}".format(top_dir, args.result_dir)):
            os.mkdir("{}/{}".format(top_dir, args.result_dir))
        elif args.result_dir != "test":
            pass

        args.result_dir = "{}/{}".format(top_dir, args.result_dir)
        fh = logging.FileHandler("{}/log-training.txt".format(args.result_dir))

        # dump args
        with Path("{}/args-train.json".format(args.result_dir)).open(
            "w", encoding="utf-8"
        ) as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt='[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    args.device = device
    logger.info(f"Using Device {device} {args.gpu_device}")
    
    args.entity_types = EntityTypes(
        args.types_path, args.negative_types_number, args.negative_mode
    )
    args.entity_types.build_types_embedding(
        args.bert_model,
        True,
        args.device,
        args.types_mode,
        args.init_type_embedding_from_bert,
    )

    if args.mt_test_only:
        if args.model_dir == "":
            raise ValueError("NULL model directory!")
        evaluate(args)
    else:
        if args.model_dir != "":
            raise ValueError("Model directory should be NULL!")
        train(args)
