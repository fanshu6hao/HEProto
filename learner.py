# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import shutil
import time
from copy import deepcopy
import pandas as pd

import numpy as np
import torch
from torch import nn

import joblib
from modeling import BertForTokenClassification_MT
from transformers import CONFIG_NAME, PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME
from transformers import AdamW as BertAdam
from transformers import get_linear_schedule_with_warmup


logger = logging.getLogger(__file__)


class Learner(nn.Module):
    ignore_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    pad_token_label_id = -1

    def __init__(
        self,
        bert_model,
        label_list,
        freeze_layer,
        logger,
        lr,
        warmup_prop,
        max_training_steps,
        model_dir="",
        cache_dir="",
        gpu_no=0,
        py_alias="python",
        args=None,
    ):
        super(Learner, self).__init__()
        self.lr = lr
        self.warmup_prop = warmup_prop
        self.max_training_steps = max_training_steps
        self.bert_model = bert_model
        self.label_list = label_list
        self.py_alias = py_alias
        self.entity_types = args.entity_types
        self.is_debug = args.debug
        self.model_dir = model_dir
        self.args = args
        self.freeze_layer = freeze_layer
        
        self.best_f1 = {"valid": 0.0, "test": 0.0}
        self.best_span_f1 = {"valid": 0.0, "test": 0.0}
        self.best_type_f1 = {"valid": 0.0, "test": 0.0}

        num_labels = len(label_list)

        if not self.args.mt_test_only:
            logger.info("********** Loading pre-trained model **********")
            cache_dir = cache_dir if cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
            self.model = BertForTokenClassification_MT.from_pretrained(
                bert_model,
                cache_dir=cache_dir,
                num_labels=num_labels,
                output_hidden_states=True,
            )

            self.model.set_config(
                args.mt_fuse_mode,
                args.mt_add_weight,
                args.type_cl_weight,
                args.fine_type_margin,
                args.fine_margin_weight,
                args.coarse_fine_cat_mode,
                args.coarse_weight,
                args.add_weight,
                args.distance_mode,
                args.ft_similar_k,
                args.span_cl_weight,
                args.span_temperature,
                args.span_scale_by_temperature,
                args.use_type_contrastive,
                args.type_temperature,
                args.type_scale_by_temperature,
            )
            self.model.to(args.device)
            self.layer_set()

    def layer_set(self):
        # layer freezing
        no_grad_param_names = ["embeddings", "pooler"] + [
            "layer.{}.".format(i) for i in range(self.freeze_layer)
        ]
        # no_grad_param_names = []
        logger.info("The frozen parameters are:")
        for name, param in self.model.named_parameters():
            if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
                param.requires_grad = False
                logger.info("  {}".format(name))

        self.opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr)  # AdamW
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(self.max_training_steps * self.warmup_prop),
            num_training_steps=self.max_training_steps,
        )

    def get_optimizer_grouped_parameters(self):
        # [bert, classifier, bert_type, mt_project_layer]
        param_optimizer = list(self.model.named_parameters())
        # for n, p in param_optimizer:
        #     logger.info(f"{n}")
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        span_paras = ['bert.', 'classifier.']
        # if self.args.multitask:
        type_paras = ['bert_type.', 'ln.', 'type_cl_project.']
        if self.args.mt_fuse_mode == 'concat':
            type_paras.append('mt_project_layer.')
        elif self.args.mt_fuse_mode == 'gate':
            type_paras.append('mt_gate.')
        elif self.args.mt_fuse_mode == 'add_auto':
            type_paras.append('mt_add_auto_weight')
            
        # if self.args.use_coarse_for_fine_proto:
        type_paras.append('bert_type_fine.')
        if self.args.coarse_fine_cat_mode == 'concat':
            type_paras.append('project_layer.')
        elif self.args.coarse_fine_cat_mode == 'gate':
            type_paras.append('gate_weight.')

        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in span_paras) and not any(nd in n for nd in no_decay) and p.requires_grad], "lr": self.args.lr_span, "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in span_paras) and any(nd in n for nd in no_decay) and p.requires_grad], "lr": self.args.lr_span, "weight_decay": 0.0},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in type_paras) and not any(nd in n for nd in no_decay) and p.requires_grad], "lr": self.args.lr_type, "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in type_paras) and any(nd in n for nd in no_decay) and p.requires_grad], "lr": self.args.lr_type, "weight_decay": 0.0},
        ]
            
        return optimizer_grouped_parameters

    def get_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names

    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params

    def load_weights(self, names, params):
        model_params = self.model.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)

    def load_gradients(self, names, grads):
        model_params = self.model.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            if model_params[n].grad is None:
                continue
            model_params[n].grad.data.add_(g.data)  # accumulate

    # def get_learning_rate(self, lr, progress, warmup, schedule="linear"):
    #     if schedule == "linear":
    #         if progress < warmup:
    #             lr *= progress / warmup
    #         else:
    #             lr *= max((progress - 1.0) / (warmup - 1.0), 0.0)
    #     return lr
    
    def forward(self, batch_query, batch_support):         
        span_losses, type_losses = [], []
        base_ce_losses, max_ce_losses, span_cl_losses = [], [], []
        coarse_type_losses, fine_type_losses, type_cl_losses = [], [], []
        fine_margin_losses = []
        total_loss = 0.0
        task_num = len(batch_query)
        
        mlw = self.args.mt_loss_weight

        self.model.train()
        # 一个个的放进模型，loss最后统一backward
        for task_id in range(task_num):
            total_loss = 0.0
            
            for key in ["input_ids", "input_mask", "segment_ids", "label_ids", "e_mask", "e_type_ids", "e_type_mask"]:
                batch_support[task_id][key] = batch_support[task_id][key].to(self.args.device)
                batch_query[task_id][key] = batch_query[task_id][key].to(self.args.device)
            
            # support set
            result = self.model.forward_wuqh(
                input_ids=batch_support[task_id]["input_ids"],
                attention_mask=batch_support[task_id]["input_mask"],
                token_type_ids=batch_support[task_id]["segment_ids"],
                labels=batch_support[task_id]["label_ids"],
                e_mask=batch_support[task_id]["e_mask"],
                e_type_ids=batch_support[task_id]["e_type_ids"],
                e_type_mask=batch_support[task_id]["e_type_mask"],
                entity_types=self.entity_types,
                is_update_type_embedding=True,  # 更新原型
                lambda_max_loss=self.args.lambda_max_loss,
                sim_k=self.args.similar_k,
            )
            span_loss, type_loss = result[2], result[3]
            if span_loss is not None:
                span_losses.append(span_loss.item())
                if mlw is not None:
                    total_loss += mlw * span_loss
                else:
                    total_loss += span_loss
                if len(result) > 4:
                    base_ce_losses.append(result[4][0].item())
                    max_ce_losses.append(result[4][1].item())
                    span_cl_losses.append(result[5].item())
            
            if type_loss is not None:
                type_losses.append(type_loss.item())
                if mlw is not None:
                    total_loss += (1 - mlw) * type_loss
                else:
                    total_loss += type_loss
                if len(result) > 4:
                    coarse_type_losses.append(result[6].item())
                    fine_type_losses.append(result[7].item())
                    # if self.args.use_fine_margin:
                    type_cl_losses.append(result[8].item())
                    fine_margin_losses.append(result[9].item())
                        
            
            total_loss = total_loss / task_num   # gradient accumulation
            total_loss.backward()
            
            total_loss = 0.0

            # query set
            result = self.model.forward_wuqh(
                input_ids=batch_query[task_id]["input_ids"],
                attention_mask=batch_query[task_id]["input_mask"],
                token_type_ids=batch_query[task_id]["segment_ids"],
                labels=batch_query[task_id]["label_ids"],
                e_mask=batch_query[task_id]["e_mask"],
                e_type_ids=batch_query[task_id]["e_type_ids"],
                e_type_mask=batch_query[task_id]["e_type_mask"],
                entity_types=self.entity_types,
                lambda_max_loss=self.args.lambda_max_loss,
            )
            span_loss, type_loss = result[2], result[3]
            if span_loss is not None:
                span_losses.append(span_loss.item())
                if mlw is not None:
                    total_loss += mlw * span_loss
                else:
                    total_loss += span_loss
                if len(result) > 4:
                    base_ce_losses.append(result[4][0].item())
                    max_ce_losses.append(result[4][1].item())
                    span_cl_losses.append(result[5].item())
            if type_loss is not None:
                type_losses.append(type_loss.item())
                if mlw is not None:
                    total_loss += (1 - mlw) * type_loss
                else:
                    total_loss += type_loss
                if len(result) > 4:
                    coarse_type_losses.append(result[6].item())
                    fine_type_losses.append(result[7].item())
                    type_cl_losses.append(result[8].item())   
                    fine_margin_losses.append(result[9].item())
            
            total_loss = total_loss / task_num
            total_loss.backward()
        
        self.opt.step()
        self.scheduler.step()
        self.opt.zero_grad()
        
        return (
            np.mean(span_losses) if span_losses else 0,
            np.mean(type_losses) if type_losses else 0,
            np.mean(base_ce_losses) if base_ce_losses else 0,
            np.mean(max_ce_losses) if max_ce_losses else 0,
            np.mean(span_cl_losses) if span_cl_losses else 0,
            np.mean(coarse_type_losses) if coarse_type_losses else 0,
            np.mean(fine_type_losses) if fine_type_losses else 0,
            np.mean(type_cl_losses) if type_cl_losses else 0,
            np.mean(fine_margin_losses) if fine_margin_losses else 0,
        )


    # ---------------------------------------- Evaluation -------------------------------------- #
    def write_result(self, words, y_true, y_pred, tmp_fn):
        assert len(y_pred) == len(y_true)
        with open(tmp_fn, "w", encoding="utf-8") as fw:
            for i, sent in enumerate(y_true):
                for j, word in enumerate(sent):
                    fw.write("{} {} {}\n".format(words[i][j], word, y_pred[i][j]))
            fw.write("\n")
    
    
    def batch_test_mt(self, data):
        N = data["input_ids"].shape[0]
        B = 16
        BATCH_KEY = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "e_mask",
            "e_type_ids",
            "e_type_mask",
        ]

        logits, e_logits, pred_e_logits, span_loss, type_loss = [], [], [], 0, 0
        e_logits_coarse, pred_e_logits_coarse = [], []
        span_results, eval_types = [], []
        for i in range((N - 1) // B + 1):
            tmp = {
                ii: jj if ii not in BATCH_KEY else jj[i * B : (i + 1) * B]
                for ii, jj in data.items()
            }
            result = self.model.evaluate_forward_wuqh(**tmp)
            tmp_l, tmp_el, tmp_span_loss, tmp_eval_type_loss = result[0], result[1], result[2], result[3]
            tmp_span_res, tmp_eval_type = result[4], result[5]
            if tmp_l is not None:
                logits.extend(tmp_l.detach().cpu().numpy())
            if tmp_el is not None:                
                e_logits_coarse.extend(tmp_el[0][0].detach().cpu().numpy()) # coarse 
                e_logits.extend(tmp_el[0][1].detach().cpu().numpy()) # fine
                pred_e_logits_coarse.extend(tmp_el[1][0].detach().cpu().numpy()) # coarse 
                pred_e_logits.extend(tmp_el[1][1].detach().cpu().numpy()) # fine
                
            if tmp_span_loss is not None:
                span_loss += tmp_span_loss
            if tmp_eval_type_loss is not None:
                type_loss += tmp_eval_type_loss
            if tmp_span_res is not None:
                span_results.extend(tmp_span_res)
            if tmp_eval_type is not None:
                eval_types.extend(tmp_eval_type)
            
        return logits, [e_logits, pred_e_logits], span_loss, type_loss, span_results, eval_types, [e_logits_coarse, pred_e_logits_coarse]
    
    
    def eval_finetune(self, data_support, lr_curr, ft_steps, viterbi_decoder = None, no_grad: bool = False):
        inner_opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=lr_curr)
        self.model.train()
        only_train_spans = True

        total_steps = ft_steps + self.args.only_ft_span_steps

        for step in range(total_steps):
            if step > self.args.only_ft_span_steps - 1:
                only_train_spans = False
                
            inner_opt.param_groups[0]["lr"] = self.args.ft_lr_span
            inner_opt.param_groups[1]["lr"] = self.args.ft_lr_span
            inner_opt.param_groups[2]["lr"] = self.args.ft_lr_type
            inner_opt.param_groups[3]["lr"] = self.args.ft_lr_type
            
            inner_opt.zero_grad()
            loss = 0

            result = self.model.forward_wuqh(
                input_ids=data_support["input_ids"],
                attention_mask=data_support["input_mask"],
                token_type_ids=data_support["segment_ids"],
                labels=data_support["label_ids"],
                e_mask=data_support["e_mask"],
                e_type_ids=data_support["e_type_ids"],
                e_type_mask=data_support["e_type_mask"],
                entity_types=self.entity_types,
                is_update_type_embedding=True,
                lambda_max_loss=self.args.ft_lambda_max_loss,
                sim_k=self.args.ft_similar_k,
                only_train_spans=only_train_spans,
            )
            span_loss, type_loss = result[2], result[3]
            
            if span_loss is not None:
                loss += span_loss
            if type_loss is not None:
                loss += type_loss
            if no_grad:
                continue
            loss.backward()
            inner_opt.step()

        return loss.item()
    
    def evaluate_mt(
        self,
        corpus,
        logger,
        lr,
        ft_steps,
        mode,
        set_type,
        viterbi_decoder=None,
        load_best_and_test=False,
    ):
        if load_best_and_test == True:
            self.load_model("mt", is_training=True)
        if self.args.mt_test_only:
            self.load_model("mt")
        
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        t_tmp = time.time()
        targets, predes, spans, type_preds, type_g = [], [], [], [], []
        predes_type, predes_e = [], []
        type_logits, type_ids = [], []
        type_preds_coarse, type_g_coarse, type_logits_coarse, type_ids_coarse = [], [], [], []

        for item_id in range(corpus.n_total):
            eval_query, eval_support = corpus.get_batch_meta(
                batch_size=1, shuffle=False
            )
            # 数据放到cuda
            for key in ["input_ids", "input_mask", "segment_ids", "label_ids", "e_mask", "e_type_ids", "e_type_mask"]:
                eval_support[0][key] = eval_support[0][key].to(self.args.device)
                eval_query[0][key] = eval_query[0][key].to(self.args.device)

            # finetune on support examples
            if not self.args.nouse_inner_ft:
                self.eval_finetune(eval_support[0], lr_curr=lr, ft_steps=ft_steps, viterbi_decoder=viterbi_decoder)

            # eval on pseudo query examples (test example)
            self.model.eval()
            with torch.no_grad():
                _, e_ls, tmp_eval_span_loss, tmp_eval_type_loss, span_results, eval_types, e_ls_coarse = self.batch_test_mt(
                    {
                        "input_ids": eval_query[0]["input_ids"],
                        "attention_mask": eval_query[0]["input_mask"],
                        "token_type_ids": eval_query[0]["segment_ids"],
                        "labels": eval_query[0]["label_ids"],
                        "e_mask": eval_query[0]["e_mask"],
                        "e_type_ids": eval_query[0]["e_type_ids"],
                        "e_type_mask": eval_query[0]["e_type_mask"],
                        "entity_types": self.entity_types,
                        "viterbi_decoder": viterbi_decoder,
                        "eval_query_types": eval_query[0]["types"],
                    }
                )
                e_logits, pred_e_logits = e_ls[0], e_ls[1]

                eval_loss += tmp_eval_span_loss
                eval_loss += tmp_eval_type_loss
                
                targets.extend(eval_query[0]["entities"])
                spans.extend(span_results)
                
                # 真实span
                type_pred, type_ground = self.eval_typing(
                    e_logits, eval_query[0]["e_type_mask"], fp_flag=True,
                )
                type_preds.append(type_pred)
                type_g.append(type_ground)
                
                e_logits_coarse, _ = e_ls_coarse[0], e_ls_coarse[1]
                
                # logger.info(f"e_logits_coarse:{e_logits_coarse, torch.tensor(e_logits_coarse).shape}")
                # logger.info(f"pred_e_logits_coarse:{pred_e_logits_coarse, torch.tensor(pred_e_logits_coarse).shape}")
                # logger.info("\n")
                
                type_pred_coarse, type_ground_coarse = self.eval_typing(
                    e_logits_coarse, eval_query[0]["e_type_mask"]
                )
                type_preds_coarse.append(type_pred_coarse)
                type_g_coarse.append(type_ground_coarse)
       
                _, p, pred_type, pred_e = self.decode_entity(
                    pred_e_logits, span_results, eval_types, eval_query[0]["entities"], need_lg=True
                )
                predes.extend(p)
                predes_type.extend(pred_type)
                predes_e.append(pred_e)
                

            nb_eval_steps += 1

            self.load_weights(names, weights)
            if item_id % 200 == 0:
                logger.info(
                    "  To sentence {}/{}. Time: {}sec".format(
                        item_id, corpus.n_total, time.time() - t_tmp
                    )
                )


        eval_loss = eval_loss / nb_eval_steps
        if self.is_debug:
            joblib.dump([targets, predes, spans], "debug/f1.pkl")
        store_dir = self.args.model_dir if self.args.model_dir else self.args.result_dir
        
        pred = [[jj[:-1] for jj in ii] for ii in predes] # predes 的格式为 [[B, E, type, 相似度/距离?], ...]
        p, r, f1 = self.cacl_f1(targets, pred)
        pred = [
            [jj[:-1] for jj in ii if jj[-1] > self.args.type_threshold] for ii in predes
        ]
        p_t, r_t, f1_t = self.cacl_f1(targets, pred)
        
        dict_type_t = self.fp_dynamic_threshold_filter(predes, self.args.dynamic_threshold_quantile)
        pred = [
            [jj[:-1] for jj in ii if jj[-1] > dict_type_t[jj[-2]]] for ii in predes
        ]
        p_t_new, r_t_new, f1_t_new = self.cacl_f1(targets, pred)
        
        # pred = [
        #     [jj[:-1] for jj in ii if jj[-2] != 67] for ii in predes
        # ]
        # p_t1, r_t1, f1_t1 = self.cacl_f1(targets, pred)
        
        # pred = [
        #     [jj[:-1] for jj in ii if jj[-1] > self.args.type_threshold and jj[-2] != 67] for ii in predes
        # ]
        # p_t2, r_t2, f1_t2 = self.cacl_f1(targets, pred)

        span_p, span_r, span_f1 = self.cacl_f1(
            [[(jj[0], jj[1]) for jj in ii] for ii in targets], spans
        )
        type_p, type_r, type_f1 = self.cacl_f1(type_g, type_preds)
        type_p_c, type_r_c, type_f1_c = self.cacl_f1(type_g_coarse, type_preds_coarse)

        results = {
            "loss": eval_loss,
            "precision": p,
            "recall": r,
            "f1": f1,
            "span_p": span_p,
            "span_r": span_r,
            "span_f1": span_f1,
            "type_p": type_p,
            "type_r": type_r,
            "type_f1": type_f1,
            "precision_threshold": p_t,
            "recall_threshold": r_t,
            "f1_threshold": f1_t,
            "precision_threshold_dynamic": p_t_new,
            "recall_threshold_dynamic": r_t_new,
            "f1_threshold_dynamic": f1_t_new,
        }
        results['type_p_coarse'] = type_p_c
        results['type_r_coarse'] = type_r_c
        results['type_f1_coarse'] = type_f1_c

        logger.info("***** Eval results %s-%s *****", mode, set_type)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info(
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
            results["precision"] * 100,
            results["recall"] * 100,
            results["f1"] * 100,
            results["span_p"] * 100,
            results["span_r"] * 100,
            results["span_f1"] * 100,
            results["type_p"] * 100,
            results["type_r"] * 100,
            results["type_f1"] * 100,
            results["precision_threshold"] * 100,
            results["recall_threshold"] * 100,
            results["f1_threshold"] * 100,
            results["precision_threshold_dynamic"] * 100,
            results["recall_threshold_dynamic"] * 100,
            results["f1_threshold_dynamic"] * 100,
        )
        
        logger.info("***********************************")
             
        # total
        if set_type == 'test':
            if not self.args.mt_test_only and (f1 > 0 and self.best_f1[set_type] < f1):
                self.best_f1[set_type] = f1
                logger.info(f"Saving {set_type} preds file to {store_dir}~~~")
                joblib.dump(
                    [targets, predes, spans, predes_type, predes_e],
                    "{}/{}_{}_preds.pkl".format(store_dir, "all", set_type),
                )
                
                # joblib.dump(
                #     mt_logits, f"{store_dir}/{set_type}_mt_logits.pkl"
                # )
                # joblib.dump(
                #     mt_labels, f"{store_dir}/{set_type}_mt_labels.pkl"
                # )
                
                logger.info(f"Saving {set_type} typing files to {store_dir}~~~")
                joblib.dump(
                    [type_g, type_preds],
                    "{}/{}_{}_preds.pkl".format(store_dir, "typing", set_type),
                )
                # joblib.dump(
                #     [type_logits, type_ids],
                #     "{}/{}_{}_preds_results.pkl".format(store_dir, "typing", set_type),
                # )
                # if self.args.use_coarse_for_fine_proto and set_type == 'test':
                joblib.dump(
                    [type_g_coarse, type_preds_coarse],
                    "{}/{}_{}_coarse_preds.pkl".format(store_dir, "typing", set_type),
                )

        return results, preds
    
    def fp_dynamic_threshold_filter(self, predes, threshold_quantile: float):
        dict_type_len = {}
        for ii in predes:
            for jj in ii:
                if jj[2] not in dict_type_len:
                    dict_type_len[jj[2]] = [jj[-1]]
                else:
                    dict_type_len[jj[2]].append(jj[-1])
                    
        dict_type_t = {}
        for key in dict_type_len:
            distance = pd.Series(dict_type_len[key])
            # stats = distance.describe()
            dict_type_t[key] = distance.quantile(threshold_quantile)
        
        return dict_type_t
            

    def save_model(self, result_dir, fn_prefix, max_seq_len, mode: str = "all"):
        # Save a trained model and the associated configuration
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        output_model_file = os.path.join(   # 保存bin文件
            result_dir, "{}_{}_{}".format(fn_prefix, mode, WEIGHTS_NAME)
        )
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(result_dir, CONFIG_NAME)   # 保存config文件
        with open(output_config_file, "w", encoding="utf-8") as f:
            f.write(model_to_save.config.to_json_string())
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        model_config = {
            "bert_model": self.bert_model,
            "do_lower": False,
            "max_seq_length": max_seq_len,
            "num_labels": len(self.label_list) + 1,
            "label_map": label_map,
        }
        json.dump(
            model_config,
            open(
                os.path.join(result_dir, f"{mode}-model_config.json"),
                "w",
                encoding="utf-8",
            ),
        )
        if mode == "type" or mode == "mt":
            joblib.dump(
                self.entity_types, os.path.join(result_dir, "type_embedding.pkl")
            )

    def save_best_model(self, result_dir: str, mode: str):
        output_model_file = os.path.join(result_dir, "en_tmp_{}".format(WEIGHTS_NAME))
        config_name = os.path.join(result_dir, "tmp-model_config.json")
        shutil.copy(output_model_file, output_model_file.replace("tmp", mode))
        shutil.copy(config_name, config_name.replace("tmp", mode))

    def load_model(self, mode: str, is_training=False):
        if not is_training:
            if not self.model_dir:
                return
            model_dir = self.model_dir
        else:
            model_dir = self.args.result_dir
        logger.info(f"********** Loading saved {mode} model **********")
        output_model_file = os.path.join(
            model_dir, "en_{}_{}".format(mode, WEIGHTS_NAME)
        )
        self.model = BertForTokenClassification_MT.from_pretrained(
            self.bert_model, num_labels=len(self.label_list), output_hidden_states=True
        )
        self.model.set_config(
                self.args.mt_fuse_mode,
                self.args.mt_add_weight,
                self.args.type_cl_weight,
                self.args.fine_type_margin,
                self.args.fine_margin_weight,
                self.args.coarse_fine_cat_mode,
                self.args.coarse_weight,
                self.args.add_weight,
                self.args.distance_mode,
                self.args.ft_similar_k,
                self.args.span_cl_weight,
                self.args.span_temperature,
                self.args.span_scale_by_temperature,
                self.args.use_type_contrastive,
                self.args.type_temperature,
                self.args.type_scale_by_temperature,
        )
        self.model.to(self.args.device)
        # 加载bin文件（模型权重）
        self.model.load_state_dict(torch.load(output_model_file, map_location="cuda"))
        self.layer_set()

    def decode_span(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        types,
        mask: torch.Tensor,
        viterbi_decoder=None,
        need_return_mask=False,
    ):
        if self.is_debug:
            joblib.dump([logits, target, self.label_list], "debug/span.pkl")
        device = target.device
        K = max([len(ii) for ii in types])
        if viterbi_decoder:
            N = target.shape[0]
            B = 16
            result = []
            for i in range((N - 1) // B + 1):
                tmp_logits = torch.tensor(logits[i * B : (i + 1) * B]).to(target.device)
                if len(tmp_logits.shape) == 2:
                    tmp_logits = tmp_logits.unsqueeze(0)
                tmp_target = target[i * B : (i + 1) * B]
                log_probs = nn.functional.log_softmax(
                    tmp_logits.detach(), dim=-1
                )  # batch_size x max_seq_len x n_labels
                pred_labels = viterbi_decoder.forward(
                    log_probs, mask[i * B : (i + 1) * B], tmp_target
                )

                for ii, jj in zip(pred_labels, tmp_target.detach().cpu().numpy()):
                    left, right, tmp = 0, 0, []
                    while right < len(jj) and jj[right] == self.ignore_token_label_id:
                        tmp.append(-1)
                        right += 1
                    while left < len(ii):
                        tmp.append(ii[left])
                        left += 1
                        right += 1
                        while (
                            right < len(jj) and jj[right] == self.ignore_token_label_id
                        ):
                            tmp.append(-1)
                            right += 1
                    result.append(tmp)
        target = target.detach().cpu().numpy()
        B, T = target.shape
        if not viterbi_decoder:
            logits = torch.tensor(logits).detach().cpu().numpy()
            result = np.argmax(logits, -1)

        if self.label_list == ["O", "B", "I"]:
            res = []
            for ii in range(B):
                tmp, idx = [], 0
                max_pad = T - 1
                while (
                    max_pad > 0 and target[ii][max_pad - 1] == self.pad_token_label_id
                ):
                    max_pad -= 1
                while idx < max_pad:
                    if target[ii][idx] == self.ignore_token_label_id or (
                        result[ii][idx] != 1 # 1是B 
                    ):
                        idx += 1
                        continue
                    e = idx
                    while e < max_pad - 1 and (
                        target[ii][e + 1] == self.ignore_token_label_id
                        or result[ii][e + 1] in [self.ignore_token_label_id, 2] # 2是I 
                    ):
                        e += 1
                    tmp.append((idx, e))
                    idx = e + 1
                res.append(tmp)
        elif self.label_list == ["O", "B", "I", "E", "S"]:
            res = []
            for ii in range(B):
                tmp, idx = [], 0
                max_pad = T - 1
                while (
                    max_pad > 0 and target[ii][max_pad - 1] == self.pad_token_label_id
                ):
                    max_pad -= 1
                while idx < max_pad:
                    if target[ii][idx] == self.ignore_token_label_id or (
                        result[ii][idx] not in [1, 4]
                    ):
                        idx += 1
                        continue
                    e = idx
                    while (
                        e < max_pad - 1
                        and result[ii][e] not in [3, 4]
                        and (
                            target[ii][e + 1] == self.ignore_token_label_id
                            or result[ii][e + 1] in [self.ignore_token_label_id, 2, 3]
                        )
                    ):
                        e += 1
                    if e < max_pad and result[ii][e] in [3, 4]: #检测实体右边界 遇到 E 或 S 时都算成 E
                        while (
                            e < max_pad - 1
                            and target[ii][e + 1] == self.ignore_token_label_id
                        ):
                            e += 1
                        tmp.append((idx, e))
                    idx = e + 1
                res.append(tmp)
        if need_return_mask:
            M = max([len(ii) for ii in res])
            e_mask = np.zeros((B, M, T), np.int8)
            e_type_mask = np.zeros((B, M, K), np.int8)
            e_type_ids = np.zeros((B, M, K), np.int_)
            for ii in range(B):
                for idx, (s, e) in enumerate(res[ii]):
                    e_mask[ii][idx][s : e + 1] = 1
                types_set = types[ii]
                if len(res[ii]):
                    e_type_ids[ii, : len(res[ii]), : len(types_set)] = [types_set] * len(
                        res[ii]
                    )
                e_type_mask[ii, : len(res[ii]), : len(types_set)] = np.ones(
                    (len(res[ii]), len(types_set))
                )
        return (
            torch.tensor(e_mask).to(device) if need_return_mask else None,
            torch.tensor(e_type_ids, dtype=torch.long).to(device) if need_return_mask else None,
            torch.tensor(e_type_mask).to(device) if need_return_mask else None,
            res,
            types,
        )

    def decode_entity(self, e_logits, result, types, entities, need_lg=False):
        if self.is_debug:
            joblib.dump([e_logits, result, types, entities], "debug/e.pkl")
        target, preds = entities, []
        res_types, res_lg = [], []
        B = len(e_logits)
        logits = e_logits

        for ii in range(B):
            tmp = []
            tmp_lg = []
            tmp_res = result[ii]
            tmp_types = types[ii]
            for jj in range(len(tmp_res)):
                lg = logits[ii][jj, : len(tmp_types)]
                if need_lg:
                    # tmp.append((*tmp_res[jj], tmp_types[np.argmax(lg[:-1])], lg[np.argmax(lg[:-1])])) # [(B, E), type, distance/similarity]
                    tmp_lg.append(lg)
                # else:
                tmp.append((*tmp_res[jj], tmp_types[np.argmax(lg)], lg[np.argmax(lg)])) # [(B, E), type, distance/similarity]
            preds.append(tmp)    
            if need_lg:
                res_types.append(tmp_types)
                res_lg.append(tmp_lg)
        if need_lg:
            return target, preds, res_types, res_lg
        else:
            return target, preds

    def cacl_f1(self, targets: list, predes: list):
        tp, fp, fn = 0, 0, 0
        for ii, jj in zip(targets, predes):
            ii, jj = set(ii), set(jj)
            same = ii - (ii - jj)
            tp += len(same)
            fn += len(ii - jj)
            fp += len(jj - ii)
        p = tp / (fp + tp + 1e-10)
        r = tp / (fn + tp + 1e-10)
        return p, r, 2 * p * r / (p + r + 1e-10)

    def eval_typing(self, e_logits, e_type_mask, fp_flag=False): # ground是全0, res是argmax的下标（正确的应该是0，因为模型计算时下标0是label对应的原型）
        e_logits = e_logits
        e_type_mask = e_type_mask.detach().cpu().numpy()
        if self.is_debug:
            joblib.dump([e_logits, e_type_mask], "debug/typing.pkl")

        N = len(e_logits)
        B_S = 16
        res = []
        for i in range((N - 1) // B_S + 1):
            tmp = e_logits[i * B_S : (i + 1) * B_S]
            tmp_e_logits = np.argmax(tmp, -1)
            B, M = tmp_e_logits.shape
            tmp_e_type_mask = e_type_mask[i * B_S : (i + 1) * B_S][:, :M, 0]
            res.extend(tmp_e_logits[tmp_e_type_mask == 1])
        ground = [0] * len(res)
        return enumerate(res), enumerate(ground)
    
    def eval_typing_results(self, e_logits, e_type_mask, e_type_ids):
        try:
            e_logits = torch.tensor(e_logits)
        except:
            # 将numpy数组转换为tensor
            tensor_list = [torch.from_numpy(logit) for logit in e_logits]
            # 对序列进行padding
            e_logits = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)
        
        e_type_mask = e_type_mask.detach().cpu()
        e_type_ids = e_type_ids.detach().cpu()
        
        _, M, _ = e_logits.shape
        e_type_mask = e_type_mask[:, :M, :]
        e_type_ids = e_type_ids[:, :M, :]
        
        type_ids = e_type_ids[e_type_mask.sum(-1)>0].tolist()
        logits = e_logits[e_type_mask.sum(-1)>0].tolist()
        
        return logits, type_ids