import argparse

def my_bool(s):
        return s != "False"

def myArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="FewNERD", help="FewNERD or Domain"
    )
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--mode", type=str, default="intra")
    # parser.add_argument(
    #     "--test_only",
    #     action="store_true",
    #     help="if true, will load the trained model and run test only",
    # )
    parser.add_argument(
        "--convert_bpe",
        action="store_true",
        help="if true, will convert the bpe encode result to word level.",
    )
    parser.add_argument("--tagging_scheme", type=str, default="BIOES", help="BIOES or BIO or IO")
    # dataset settings
    parser.add_argument("--data_path", type=str, default="./dataset/episode-data")
    parser.add_argument(
        "--result_dir", type=str, help="where to save the result.", default="test"
    )
    parser.add_argument(
        "--model_dir", type=str, help="dir name of a trained model", default=""
    )

    # test setting
    parser.add_argument(
        "--lr_finetune",
        type=float,
        help="finetune learning rate, used in [test_meta]. and [k_shot setting]",
        default=1e-5,
    )
    parser.add_argument(
        "--lr_finetune_type",
        type=float,
        help="finetune learning rate, used in [test_meta]. and [k_shot setting]",
        default=1e-5,
    )
    parser.add_argument(
        "--max_ft_steps", type=int, default=3
    )
    # parser.add_argument(
    #     "--max_type_ft_steps",
    #     type=int,
    #     help="maximal steps token for entity type fine-tune.",
    #     default=3,
    # )

    # train setting
    parser.add_argument(
        "--batch_size",
        type=int,
        help="[number of tasks] batch_size",
        default=32,
    )
    parser.add_argument(
        "--lr", type=float, help="learning rate", default=1e-5
    )
    parser.add_argument(
        "--max_training_steps",
        type=int,
        help="maximal steps token for training.",
        default=5001,
    )
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument(
        "--warmup_prop",
        type=float,
        help="warm up proportion for meta update",
        default=0.1,
    )

    # permanent params
    parser.add_argument(
        "--freeze_layer", type=int, help="the layer of mBERT to be frozen", default=0
    )
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--bert_model",
        type=str,
        default="../model/bert-base-uncased/",
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
        default="",
    )
    parser.add_argument(
        "--viterbi", type=str, default="hard", help="hard or soft or None"
    )
    parser.add_argument(
        "--concat_types", type=str, default="None", help="past or before or None"
    )
    # expt setting
    parser.add_argument(
        "--seed", type=int, help="random seed to reproduce the result.", default=667
    )
    parser.add_argument("--gpu_device", type=str, help="GPU device num", default="0")
    parser.add_argument("--py_alias", type=str, help="python alias", default="python")
    parser.add_argument(
        "--types_path",
        type=str,
        help="the path of entities types",
        default="data/entity_types_v2.json",
    )
    parser.add_argument(
        "--negative_types_number",
        type=int,
        help="the number of negative types in each batch",
        default=4,
    )
    parser.add_argument(
        "--negative_mode", type=str, help="the mode of negative types", default="batch"
    )
    parser.add_argument(
        "--types_mode", type=str, help="the embedding mode of type span", default="cls"
    )
    parser.add_argument("--name", type=str, help="the name of experiment", default="")
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument(
        "--init_type_embedding_from_bert",
        action="store_true",
        help="initialization type embedding from BERT",
    )

    parser.add_argument(
        "--use_classify",
        action="store_true",
        help="use classifier after entity embedding",
    )
    parser.add_argument(
        "--distance_mode", type=str, help="embedding distance mode", default="cos"
    )
    parser.add_argument("--similar_k", type=float, help="cosine similar k", default=10)
    # parser.add_argument("--train_mode", default="add", type=str, help="add, span, type")
    # parser.add_argument("--eval_mode", default="add", type=str, help="add, two-stage")
    parser.add_argument(
        "--type_threshold", default=2.5, type=float, help="typing decoder threshold"
    )
    parser.add_argument(
        "--lambda_max_loss", default=2.0, type=float, help="span max loss lambda"
    )
    parser.add_argument(
        "--ft_lambda_max_loss", default=5.0, type=float, help="span max loss lambda"
    )
    parser.add_argument(
        "--ft_similar_k", type=float, help="cosine similar k", default=10
    )
    parser.add_argument(
        "--ignore_eval_test", help="if/not eval in test", action="store_true"
    )
    parser.add_argument(
        "--load_best_and_test", type=bool, default=False, help="训练结束时加载best model 进行 test"
    )
    parser.add_argument(
        "--nouse_inner_ft",
        action="store_true",
        help="if true, will convert the bpe encode result to word level.",
    )
    parser.add_argument(
        "--only_ft_span_steps", type=int, default=0, help="only train span steps"
    )

    parser.add_argument(
        "--span_cl_weight", type=float, default=0.1, help="weight of the span contrastive loss"
    )
    parser.add_argument(
        "--span_temperature",
        type=float,
        default=0.1,
        help="the temperature parameter for span contrastive loss"
    )
    parser.add_argument(
        "--span_scale_by_temperature",
        type=bool,
        default=False,
        help="if true, contrastive loss *= temperature"
    )
    parser.add_argument(
        "--type_temperature",
        type=float,
        default=0.1,
        help="the temperature parameter for type contrastive loss"
    )
    parser.add_argument(
        "--type_scale_by_temperature",
        type=bool,
        default=False,
        help="if true, contrastive loss *= temperature"
    )
    parser.add_argument(
        "--use_type_contrastive", type=str, help="type contrastive mode, none or token or entity", default="token"
    )
    parser.add_argument(
        "--type_cl_weight", type=float, default=0.1, help="weight of type contrastive loss"
    )
    
    parser.add_argument(
        "--fine_margin_weight", type=float, default=0.1, help="similarity margin loss weight"
    )
    parser.add_argument(
        "--fine_type_margin", type=float, default=4.0,
    )
    parser.add_argument(
        "--coarse_fine_cat_mode", type=str, default="gate", help="concat, add, add_auto, gate"
    )
    parser.add_argument(
        "--coarse_weight", type=float, default=1.0, help="coarse type loss weight"
    )
    parser.add_argument(
        "--add_weight", type=float, default=None, help="add_weight * coarse_embedding + (1-add_weight) * fine_embedding"
    )
    parser.add_argument(
        "--mt_fuse_mode", type=str, default="add_auto", help="span type fuse mode"
    )
    parser.add_argument(
        "--mt_add_weight", type=float, default=0.1, help="if fuse mode is add, type = add weight * span + (1-add_weight) * type"
    )
    parser.add_argument(
        "--mt_loss_weight", type=float, default=None, help="mt loss weight * span loss + (1 - mt loss weight) * type loss"
    )
    
    parser.add_argument(
        "--lr_span", type=float, default=None, help="span lr"
    )
    parser.add_argument(
        "--lr_type", type=float, default=None, help="type lr"
    )
    parser.add_argument(
        "--ft_lr_span", type=float, default=None, help="ft span lr"
    )
    parser.add_argument(
        "--ft_lr_type", type=float, default=None, help="ft type lr"
    )
    parser.add_argument(
        "--mt_test_only", action="store_true", help="multitask test only"
    )
    parser.add_argument(
        "--mt_model_dir", type=str, default=None, help="multitask model directory for mt test only"
    )
    parser.add_argument(
        "--dynamic_threshold_quantile", type=float, default=0.1
    )

    args = parser.parse_args()
    args.negative_types_number = args.N - 1
    
    if args.mode == 'intra':
        args.dynamic_threshold_quantile = 0.15
        args.fine_type_margin = 2.0
        args.only_ft_span_steps = 27
    
    if args.mt_test_only:
        args.lr_span = args.ft_lr_span
        # args.lr_classifier = args.ft_lr_classifier
        args.lr_type = args.ft_lr_type
    else:
        if args.freeze_layer > 0:
            args.name += f"_freeze{args.freeze_layer}"
        
        # if args.mt_fuse_mode != None:
        #     args.name += f"_{args.mt_fuse_mode}"
        # if args.mt_add_weight != None:
        #     args.name += f"_aw{args.mt_add_weight}"
            
        if args.mt_loss_weight != None:
            args.name += f"_mtlw{args.mt_loss_weight}"
        
        # if args.span_cl_weight != 1.0:
        #     args.name += f"_clw{args.span_cl_weight}"
            
        # if args.use_type_contrastive != 'none':
        #     args.name += f"_{args.use_type_contrastive}{args.type_cl_weight}"

        # args.name += f'_{args.coarse_fine_cat_mode}_cw{args.coarse_weight}'
        # if args.coarse_fine_cat_mode == 'add' and args.add_weight != None:
        #     args.name += f'_aw{args.add_weight}'

        # args.name += f'_fmw{args.fine_margin_weight}'
        if args.fine_type_margin > 0:
            args.name += f'_m{args.fine_type_margin}'
           
    return args