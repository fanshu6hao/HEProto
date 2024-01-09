import logging
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np

from transformers import BertForTokenClassification
from utils import load_file


logger = logging.getLogger(__file__)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # logger.info(f"labels:{labels}")
        # logger.info(f"fetures.shape={features.shape}")
        ignore_label = (labels != 0).float()
        ignore_mask = ignore_label.repeat(ignore_label.shape[0], 1)
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)  # 变成 [batch_size, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # logger.info(f"mask:{mask}")
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask 
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        positives_mask *= ignore_mask  # 去掉 0 label相互之间的正样本
        negatives_mask = 1. - mask
        # logger.info(f"logits_mask:{logits_mask, logits_mask.shape}")
        # logger.info(f"positives_mask:{positives_mask, positives_mask.shape}")
        # logger.info(f"negatives_mask:{negatives_mask, negatives_mask.shape}")
        del mask, logits_mask, ignore_mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''        
        num_positives_per_row = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]
        # num_negatives_per_row = torch.sum(negatives_mask , axis=1) # 除了自己之外，负样本的个数  [2 0 2 2]
        # logger.info(f"num_positives_per_row = {num_positives_per_row}")
        # logger.info(f"num_negatives_per_row = {num_negatives_per_row}")       
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
                exp_logits * positives_mask, axis=1, keepdims=True)  
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        
        
        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
            
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class BertForTokenClassification_MT(BertForTokenClassification):
    def __init__(self, *args, **kwargs):
        super(BertForTokenClassification_MT, self).__init__(*args, **kwargs)
        self.input_size = 768
        self.span_loss = nn.functional.cross_entropy
        self.type_loss = nn.functional.cross_entropy
        self.dropout = nn.Dropout(p=0.1)
        # self.log_softmax = nn.functional.log_softmax

    def set_config(
        self,
        mt_fuse_mode: str = None,
        mt_add_weight: float = None,
        # mt_classifier_weight: float = None,
        type_cl_weight: float = None,
        # use_coarse_for_fine_proto: bool = False,
        # use_fine_margin: bool = False, # 用粗粒度原型 和 细粒度embedding 计算loss
        fine_type_margin: float = 0,
        fine_margin_weight: float = None,
        coarse_fine_cat_mode: str = 'concat',
        coarse_weight: float = 1.0,
        add_weight:float = None,
        distance_mode: str = "cos",
        similar_k: float = 10,
        span_cl_weight: float = 1.0,
        span_temperature: float = 0.1,
        span_scale_by_temperature: bool = False,
        use_type_contrastive: str = "none",
        type_temperature: float = 0.1,
        type_scale_by_temperature: bool = False,
    ):
        self.bert_type = deepcopy(self.bert)
        self.proto_size = 768
        self.ln = nn.LayerNorm(self.proto_size, 1e-5, True)
        
        self.mt_fuse_mode = mt_fuse_mode
        self.mt_add_weight = mt_add_weight
        
        # self.use_coarse_for_fine_proto = use_coarse_for_fine_proto
        # self.use_fine_margin = use_fine_margin
        self.coarse_fine_cat_mode = coarse_fine_cat_mode
        self.coarse_weight = coarse_weight
        self.distance_mode = distance_mode
        self.similar_k = similar_k
        
        self.span_cl_weight = span_cl_weight
        self.span_contrastive_loss = SupConLoss(temperature=span_temperature, scale_by_temperature=span_scale_by_temperature)
        self.type_contrastive_loss = SupConLoss(temperature=type_temperature, scale_by_temperature=type_scale_by_temperature)
        self.use_type_contrastive = use_type_contrastive
        
        self.type_cl_weight = type_cl_weight
        
        # self.span_cl_project = nn.Linear(768, 256)
        self.type_cl_project = nn.Linear(768, 256)
        
        config = {
            "fine_type_margin": fine_type_margin,
            "fine_margin_weight": fine_margin_weight,
            "coarse_fine_cat_mode": coarse_fine_cat_mode,
            "coarse_weight": coarse_weight,
            "add_weight": add_weight,
            "distance_mode": distance_mode,
            "similar_k": similar_k,
        }
        logger.info(f"Model Setting: {config}")
        config = {
            "span_cl_weight": span_cl_weight,
            "span_temperature": span_temperature,
            "span_scale_by_temperature": span_scale_by_temperature,
            "use_type_contrastive": use_type_contrastive,
            "type_cl_weight": type_cl_weight,
            "type_temperature": type_temperature,
            "type_scale_by_temperature": type_scale_by_temperature,
        }
        logger.info(f"Model Setting: {config}")
        
        if self.mt_fuse_mode == 'concat':
            self.mt_project_layer = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size)
            )
        elif self.mt_fuse_mode == 'gate':
            self.mt_gate = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.Sigmoid()
            )
        elif self.mt_fuse_mode == 'add_auto':
            self.mt_add_auto_weight = nn.Parameter(torch.FloatTensor([self.mt_add_weight]), requires_grad=True)
            
        # if self.use_coarse_for_fine_proto: # 粗粒度 和 细粒度 都用
        self.fine2coarse_map = self.build_types_map()
        self.bert_type_fine = deepcopy(self.bert)
        
        # if self.use_fine_margin:
        self.fine_margin_weight = fine_margin_weight
        self.fine_type_margin = fine_type_margin
        
        if self.coarse_fine_cat_mode == 'concat':
            self.project_layer = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size)
            )
        elif self.coarse_fine_cat_mode == 'add':
            self.add_weight = add_weight
        elif self.coarse_fine_cat_mode =='add_auto':
            self.add_auto_weight = nn.Parameter(torch.FloatTensor([1, 1]), requires_grad=True)
        elif self.coarse_fine_cat_mode == 'gate':
            self.gate_weight = nn.Sequential(
                nn.Linear(self.input_size * 2, self.input_size),
                nn.Sigmoid()
            )
            

    def build_types_map(self):
        self.types = load_file("data/entity_types_v2.json", "json")
        types_list = [jj for ii in self.types.values() for jj in ii]
        # 构造粗粒度types map
        coarse_types_list = types_list[-8:]
        coarse_types_map = {jj: ii+len(types_list)-8 for ii, jj in enumerate(coarse_types_list)}
        coarse_types_map['O'] = 0

        # 细粒度
        types_list = sorted(types_list[:-8])
        types_map = {jj: ii for ii, jj in enumerate(types_list)}
        return {v: coarse_types_map[k.split('-')[0]] for k,v in types_map.items()}
            
    def covert_fine2coarse(self, types_id, types_mask):
        types_id = types_id.clone()
        types_mask = types_mask.clone()
        # 将细粒度的label转换成粗粒度的label
        # 将dict转换成Tensor
        label_dict = torch.tensor(list(self.fine2coarse_map.values())).to(types_id.device)

        # 将types_id中的细粒度标签转换为粗粒度标签
        coarse_types_id = label_dict[types_id]
        
        # sorting the rows so that duplicate values appear together
        # e.g., first row: [1, 2, 3, 3, 3, 4, 4]
        y, indices = coarse_types_id.sort(dim=-1, stable=True)

        # subtracting, so duplicate values will become 0
        # e.g., first row: [1, 2, 3, 0, 0, 4, 0]
        y[:, :, 1:] *= ((y[:, :, 1:] - y[:, :, :-1]) !=0).long()

        # retrieving the original indices of elements
        indices = indices.sort(dim=-1)[1]

        # re-organizing the rows following original order
        # e.g., first row: [1, 2, 3, 4, 0, 0, 0]
        
        # logger.info(f"indices = {indices}")
        
        coarse_types_id = y.gather(-1, indices)
        types_mask[coarse_types_id == 0] = 0
        
        # logger.info(f"y = {y, y.shape}")
        # logger.info(f"coarse_types_id: {coarse_types_id, coarse_types_id.shape}")
        # logger.info(f"types_mask: {types_mask, types_mask.shape}\n")

        return coarse_types_id, types_mask
        
    def forward_wuqh(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        e_mask=None,
        e_type_ids=None,
        e_type_mask=None,
        entity_types=None,
        entity_mode: str = "mean",
        is_update_type_embedding: bool = False,
        lambda_max_loss: float = 0.0,
        sim_k: float = 0,
        only_train_spans: bool = False, # 只训练span模型
    ):
        max_len = (attention_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len].type(torch.int8)
        token_type_ids = token_type_ids[:, :max_len]
        labels = labels[:, :max_len]  # batch_size × seq_len
        
        # tmp_e_mask = torch.sum(e_mask, dim=1)[:, :max_len]
        
        # logger.info(f"e_mask = {tmp_e_mask, tmp_e_mask.shape}")
        # logger.info(f"labels = {labels, labels.shape}")
        
        # span
        span_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        span_sequence_output = self.dropout(span_output[0])  # batch size × seq_len × hidden_size
        
        # tmp_span = span_sequence_output.clone()
        # tmp_span[tmp_e_mask == 1]= 0.0
        # tmp_span[labels == -1] = 0.0
        # logger.info(f"tmp_span = {tmp_span}")
        # logger.info(f"span_sequence_output = {span_sequence_output}")
        
        # logger.info(f"\n")
        
        
        if not only_train_spans: 
            # type
            type_output = self.bert_type(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            type_sequence_output = self.dropout(type_output[0])  # batch size × seq_len × hidden_size
            
            del type_output
            
            # span + type
            if self.mt_fuse_mode == 'concat':
                type_sequence_output = torch.cat((span_sequence_output, type_sequence_output), dim=2)
                type_sequence_output = self.mt_project_layer(type_sequence_output)
            elif self.mt_fuse_mode == 'add':
                if self.mt_add_weight != None:
                    type_sequence_output = self.mt_add_weight * span_sequence_output + type_sequence_output
                else:
                    type_sequence_output += span_sequence_output
            elif self.mt_fuse_mode == 'gate':
                gate = self.mt_gate(torch.cat((span_sequence_output, type_sequence_output), dim=2))
                type_sequence_output_fine = gate * span_sequence_output + (1-gate) * type_sequence_output
                # logger.info(f"Gate = {gate, gate.shape}")
            elif self.mt_fuse_mode == 'add_auto':
                type_sequence_output = self.mt_add_auto_weight[0] * span_sequence_output + type_sequence_output
                # logger.info(f"auto weight = {self.mt_add_auto_weight[0]}")


            # type coarse + fine
            type_output_fine = self.bert_type_fine(input_ids, attention_mask, token_type_ids)
            type_sequence_output_fine = self.dropout(type_output_fine[0])
            # logger.info(f"type_sequence_output_fine1.shape: {type_sequence_output_fine.shape}")
            
            del type_output_fine
            
            if self.coarse_fine_cat_mode == 'concat':
                type_sequence_output_fine = torch.cat((type_sequence_output, type_sequence_output_fine), dim=2)
                # logger.info(f"type_sequence_output_fine2.shape: {type_sequence_output_fine.shape}")
                type_sequence_output_fine = self.project_layer(type_sequence_output_fine)
                # logger.info(f"type_sequence_output_fine3.shape: {type_sequence_output_fine.shape}")
            
            elif self.coarse_fine_cat_mode == 'add':
                if self.add_weight != None:
                    type_sequence_output_fine = self.add_weight * type_sequence_output + (1 - self.add_weight) * type_sequence_output_fine
                else:
                    type_sequence_output_fine += type_sequence_output
                # type_sequence_output_fine = F.normalize(type_sequence_output_fine, p=2, dim=-1) # 使用L2范数进行归一化
            
            elif self.coarse_fine_cat_mode == 'add_auto':
                weights = torch.nn.functional.softmax(self.add_auto_weight, dim=0)
                type_sequence_output_fine = weights[0] * type_sequence_output + weights[1] * type_sequence_output_fine
            
            elif self.coarse_fine_cat_mode == 'gate':
                gate = self.gate_weight(torch.cat((type_sequence_output, type_sequence_output_fine), dim=2))
                type_sequence_output_fine = gate * type_sequence_output + (1-gate) * type_sequence_output_fine
                
                      
        span_logits = self.classifier(span_sequence_output)  # batch_size x seq_len x num_labels    span 中 num_labels 为 5, BIOES
        
        loss_fct = CrossEntropyLoss(reduction="none")
        total_span_loss = 0
        B, M, T = span_logits.shape
        if attention_mask is not None:
            # logger.info(f"attention mask:{attention_mask}")
            active_loss = attention_mask.view(-1) == 1
            # logger.info(f"active loss:{active_loss}")
            active_logits = span_logits.reshape(-1, self.num_labels)[active_loss]
            active_seq_outputs = span_sequence_output.reshape(B*M, -1)[active_loss]
            # logger.info(f"active logits:{active_logits}")
            # logger.info(f"active logits.shape: {active_logits.shape}")  # [batch_size × 有效seq_len, num_labels]    这里的num_labels应该是 BIOES ,值为5
            active_labels = labels.reshape(-1)[active_loss]
            # logger.info(f"active labels:{active_labels}")
            # logger.info(f"active_labels.shape:{active_labels.shape}\n")  # [batch_size × 有效seq_len]
            span_contrastive_loss = self.calc_span_contrastive_loss(active_seq_outputs, active_labels)
            if torch.isnan(span_contrastive_loss):
                span_contrastive_loss = torch.tensor(0)
            #     logger.info(f"active_seq_outputs.shape:{active_seq_outputs.shape}")
            # logger.info(f"active_labels: {active_labels, active_labels.shape}")
            base_loss = loss_fct(active_logits, active_labels)
            base_ce_loss = torch.mean(base_loss)

            # max-loss
            if lambda_max_loss > 0:
                active_loss = active_loss.view(B, M)
                active_max = []
                start_id = 0
                for i in range(B):
                    sent_len = torch.sum(active_loss[i])
                    end_id = start_id + sent_len
                    active_max.append(torch.max(base_loss[start_id:end_id]))
                    start_id = end_id

                max_ce_loss = lambda_max_loss * torch.mean(torch.stack(active_max))
                span_ce_loss = base_ce_loss + max_ce_loss
            else:
                max_ce_loss = torch.tensor(0)
                span_ce_loss = base_ce_loss     
        else:
            raise ValueError("Miss attention mask!")
        
        total_span_loss = span_ce_loss + self.span_cl_weight * span_contrastive_loss
        
        if only_train_spans:
            return span_logits, None, total_span_loss, None, [base_ce_loss, max_ce_loss], span_contrastive_loss, None, None, None
        
        
        e_logits, total_type_loss, coarse_type_loss, fine_type_loss, type_cl_loss, fine_type_margin_loss = self.train_proto(
            e_mask, 
            e_type_ids, 
            e_type_mask, 
            entity_types, 
            entity_mode, 
            is_update_type_embedding, 
            sim_k, 
            max_len, 
            type_sequence_output, 
            type_sequence_output_fine
        )
        return span_logits, e_logits, total_span_loss, total_type_loss, [base_ce_loss, max_ce_loss], span_contrastive_loss, coarse_type_loss, fine_type_loss, type_cl_loss, fine_type_margin_loss

    def train_proto(self, e_mask, e_type_ids, e_type_mask, entity_types, entity_mode, is_update_type_embedding, sim_k, max_len, type_sequence_output, type_sequence_output_fine):
        # if self.use_coarse_for_fine_proto: 
        total_type_loss = 0
        coarse_type_contrastive_loss, fine_type_contrastive_loss = torch.tensor(0), torch.tensor(0)
        if e_type_mask.sum() != 0:
            M = (e_type_mask[:, :, 0] != 0).max(0)[0].nonzero(as_tuple=False)[
                -1
            ].item() + 1
        else:
            M = 1
        
        e_mask = e_mask[:, :M, :max_len].type(torch.int8)  # batch_size x max_entity_num x maxlen
        e_type_ids = e_type_ids[:, :M, :]  # batch_size x max_entity_num x K
        e_type_mask = e_type_mask[:, :M, :].type(torch.int8)  # batch_size x max_entity_num x K
    
        # 细粒度label转化成粗粒度label
        coarse_e_type_ids, coarse_e_type_mask = self.covert_fine2coarse(e_type_ids, e_type_mask)
            
        ### 使用 token 构造对比学习loss
        if self.use_type_contrastive == 'token' and torch.sum(e_mask) > 1:
            coarse_type_contrastive_loss = self.calc_type_token_contrastive_loss(
                type_sequence_output,
                e_mask, 
                coarse_e_type_ids,
            )
            fine_type_contrastive_loss = self.calc_type_token_contrastive_loss(
                type_sequence_output_fine,
                e_mask, 
                e_type_ids,
            )
        
        B, M, K = e_type_ids.shape
        
        coarse_e_out = self.get_enity_hidden(
            type_sequence_output,
            e_mask,
            entity_mode,
        )
        fine_e_out = self.get_enity_hidden(
            type_sequence_output_fine,
            e_mask,
            entity_mode
        )
            
        coarse_e_out = self.ln(coarse_e_out)  # batch_size x max_entity_num x hidden_size
        fine_e_out = self.ln(fine_e_out)
        
        if is_update_type_embedding:
            entity_types.update_type_embedding(coarse_e_out, coarse_e_type_ids, coarse_e_type_mask)   # support set时运行这里，更新原型embedding
            entity_types.update_type_embedding(fine_e_out, e_type_ids, e_type_mask)   # support set时运行这里，更新原型embedding
            
        
        ### 用 entity 的 embedding 计算对比学习
        if self.use_type_contrastive == 'entity':
            coarse_type_contrastive_loss = self.calc_type_entity_contrastive_loss(coarse_e_out, coarse_e_type_ids, coarse_e_type_mask)
            fine_type_contrastive_loss = self.calc_type_entity_contrastive_loss(fine_e_out, e_type_ids, e_type_mask)
            
        coarse_e_out = coarse_e_out.unsqueeze(2).expand(B, M, K, -1)
        coarse_types = self.get_types_embedding(
            coarse_e_type_ids, entity_types
        )  # batch_size x max_entity_num x K x hidden_size
        
        fine_e_out = fine_e_out.unsqueeze(2).expand(B, M, K, -1)
        fine_types = self.get_types_embedding(
            e_type_ids, entity_types
        )

        sim_k = sim_k if sim_k else self.similar_k
        coarse_e_types = sim_k * (coarse_e_out * coarse_types).sum(-1) / coarse_types.shape[-1]  # batch_size x max_entity_num x K
        
        fine_e_types = sim_k * (fine_e_out * fine_types).sum(-1) / fine_types.shape[-1]  # batch_size x max_entity_num x K
        
        ##### 用 fine embedding 和 coarse prototypes
        new_fine_e_types = sim_k * (fine_e_out * coarse_types).sum(-1) / coarse_types.shape[-1]
        
        # coarse_e_types[coarse_e_type_mask == 0] = 0.0 # 防止计算loss的时候将正样本当做负样本
            
        coarse_e_logits = coarse_e_types
        fine_e_logits = fine_e_types
        e_logits = [coarse_e_logits, fine_e_logits] # 方便返回

        if M:
            e_type_label = torch.zeros((B, M)).to(coarse_e_types.device)   # 全0

            ##### coarse    ce
            coarse_em = coarse_e_type_mask.clone()
            coarse_em[coarse_em.sum(-1) == 0] = 1
            
            coarse_e = coarse_e_types * coarse_em   # 距离/相似度   
    
            coarse_type_loss = self.calc_loss(
                self.type_loss, coarse_e, e_type_label, coarse_e_type_mask[:, :, 0]   # e_type_mask[:, :, 0] 对应的是有效 label, 即最后一维中的 0 号位置的元素
            )
            
            ##### fine      ce
            fine_em = e_type_mask.clone()
            fine_em[fine_em.sum(-1) == 0] = 1
            
            fine_e = fine_e_types * fine_em   # 距离/相似度

            fine_type_loss = self.calc_loss(
                self.type_loss, fine_e, e_type_label, e_type_mask[:, :, 0]   # e_type_mask[:, :, 0] 对应的是有效 label, 即最后一维中的 0 号位置的元素
            )
            
            ##### 用 fine embedding 和 coarse prototypes
            new_fine_e = new_fine_e_types * coarse_em
            fine_type_margin_loss = self.calc_fine_margin_loss(coarse_e_type_mask, new_fine_e)
                            
        else:
            coarse_type_loss = torch.tensor(0).to(type_sequence_output.device)
            fine_type_loss = torch.tensor(0).to(type_sequence_output_fine.device)
            fine_type_margin_loss = torch.tensor(0).to(type_sequence_output_fine.device)
            type_contrastive_loss = torch.tensor(0)
            
        type_contrastive_loss = coarse_type_contrastive_loss + fine_type_contrastive_loss
        
        total_type_loss = self.coarse_weight * coarse_type_loss + fine_type_loss + self.fine_margin_weight * fine_type_margin_loss + self.type_cl_weight * type_contrastive_loss
        
        return e_logits, total_type_loss, coarse_type_loss, fine_type_loss, type_contrastive_loss, fine_type_margin_loss
    

################################################################
########################### evaluate ###########################
################################################################
    def evaluate_forward_wuqh(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        e_mask=None,
        e_type_ids=None,
        e_type_mask=None,
        entity_types=None,
        viterbi_decoder=None,
        eval_query_types=None,
        entity_mode: str = "mean",
        is_update_type_embedding: bool = False,
        lambda_max_loss: float = 0.0,
        sim_k: float = 0,
    ):
        max_len = (attention_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len].type(torch.int8)
        token_type_ids = token_type_ids[:, :max_len]
        labels = labels[:, :max_len]  # batch_size × seq_len
               
        # span
        span_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        span_sequence_output = self.dropout(span_output[0])  # batch size × seq_len × hidden_size
        
        span_logits = self.classifier(span_sequence_output)  # batch_size x seq_len x num_labels    span 中 num_labels 为 5, BIOES
        
        # type
        type_output = self.bert_type(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        type_sequence_output = self.dropout(type_output[0])  # batch size × seq_len × hidden_size
            
        del span_output, type_output
       
        # span + type
        if self.mt_fuse_mode == 'concat':
            type_sequence_output = torch.cat((span_sequence_output, type_sequence_output), dim=2)
            type_sequence_output = self.mt_project_layer(type_sequence_output)
        elif self.mt_fuse_mode == 'add':
            if self.mt_add_weight != None:
                type_sequence_output = self.mt_add_weight * span_sequence_output + type_sequence_output
            else:
                type_sequence_output += span_sequence_output
        elif self.mt_fuse_mode == 'gate':
            gate = self.mt_gate(torch.cat((span_sequence_output, type_sequence_output), dim=2))
            type_sequence_output_fine = gate * span_sequence_output + (1-gate) * type_sequence_output
        elif self.mt_fuse_mode == 'add_auto':
            type_sequence_output = self.mt_add_auto_weight[0] * span_sequence_output + type_sequence_output
      
        # type coarse + fine
        type_output_fine = self.bert_type_fine(input_ids, attention_mask, token_type_ids)
        type_sequence_output_fine = self.dropout(type_output_fine[0])
        # logger.info(f"type_sequence_output_fine1.shape: {type_sequence_output_fine.shape}")
        
        del type_output_fine
        
        if self.coarse_fine_cat_mode == 'concat':
            type_sequence_output_fine = torch.cat((type_sequence_output, type_sequence_output_fine), dim=2)
            # logger.info(f"type_sequence_output_fine2.shape: {type_sequence_output_fine.shape}")
            type_sequence_output_fine = self.project_layer(type_sequence_output_fine)
            # logger.info(f"type_sequence_output_fine3.shape: {type_sequence_output_fine.shape}")
        
        elif self.coarse_fine_cat_mode == 'add':
            if self.add_weight != None:
                type_sequence_output_fine = self.add_weight * type_sequence_output + (1 - self.add_weight) * type_sequence_output_fine
            else:
                type_sequence_output_fine += type_sequence_output
            # type_sequence_output_fine = F.normalize(type_sequence_output_fine, p=2, dim=-1) # 使用L2范数进行归一化
        
        elif self.coarse_fine_cat_mode == 'add_auto':
            weights = torch.nn.functional.softmax(self.add_auto_weight, dim=0)
            type_sequence_output_fine = weights[0] * type_sequence_output + weights[1] * type_sequence_output_fine
        
        elif self.coarse_fine_cat_mode == 'gate':
            gate = self.gate_weight(torch.cat((type_sequence_output, type_sequence_output_fine), dim=2))
            type_sequence_output_fine = gate * type_sequence_output + (1-gate) * type_sequence_output_fine
        
        # Only keep active parts of the loss    # 打mask的时候padding的部分不算，active部分指的是原始句子
        loss_fct = CrossEntropyLoss(reduction="none")
        total_span_loss = 0
        B, M, T = span_logits.shape
        if attention_mask is not None:
            # logger.info(f"attention mask:{attention_mask}")
            active_loss = attention_mask.view(-1) == 1
            # logger.info(f"active loss:{active_loss}")
            active_logits = span_logits.reshape(-1, self.num_labels)[active_loss]
            active_seq_outputs = span_sequence_output.reshape(B*M, -1)[active_loss]
            # logger.info(f"active logits:{active_logits}")
            # logger.info(f"active logits.shape{active_logits.shape}")  # [batch_size × 有效seq_len, num_labels]    这里的num_labels应该是 BIOES ,值为5
            active_labels = labels.reshape(-1)[active_loss]
            # logger.info(f"active labels:{active_labels}")
            # logger.info(f"active_labels.shape:{active_labels.shape}\n")  # [batch_size × 有效seq_len]
            span_contrastive_loss = self.calc_span_contrastive_loss(active_seq_outputs, active_labels)
            if torch.isnan(span_contrastive_loss):
                span_contrastive_loss = torch.tensor(0)
            #     logger.info(f"active_seq_outputs.shape:{active_seq_outputs.shape}")
            #     logger.info(f"active_labels:{active_labels, active_labels.shape}")
            base_loss = loss_fct(active_logits, active_labels)
            base_ce_loss = torch.mean(base_loss)

            # max-loss
            if lambda_max_loss > 0:
                active_loss = active_loss.view(B, M)
                active_max = []
                start_id = 0
                for i in range(B):
                    sent_len = torch.sum(active_loss[i])
                    end_id = start_id + sent_len
                    active_max.append(torch.max(base_loss[start_id:end_id]))
                    start_id = end_id

                max_ce_loss = lambda_max_loss * torch.mean(torch.stack(active_max))
                span_ce_loss = base_ce_loss + max_ce_loss
            else:
                max_ce_loss = torch.tensor(0)
                span_ce_loss = base_ce_loss     
        else:
            raise ValueError("Miss attention mask!")
        
        total_span_loss = span_ce_loss + self.span_cl_weight * span_contrastive_loss
        
        # 使用预测的span, 用于后续分类
        pred_e_mask, pred_e_type_ids, pred_e_type_mask, span_results, eval_types = self.decode_span(
            span_logits,
            labels,
            eval_query_types,
            attention_mask,
            viterbi_decoder,
        )
        
        # 使用真实span
        _, e_logits = self.evaluate_proto(
            e_mask, 
            e_type_ids, 
            e_type_mask, 
            entity_types, 
            entity_mode, 
            is_update_type_embedding, 
            sim_k, 
            max_len, 
            type_sequence_output, 
            type_sequence_output_fine
        )
        
        # 使用预测的span
        pred_total_type_loss, pred_e_logits = self.evaluate_proto(
            pred_e_mask, 
            pred_e_type_ids, 
            pred_e_type_mask, 
            entity_types, 
            entity_mode, 
            is_update_type_embedding, 
            sim_k, 
            max_len, 
            type_sequence_output, 
            type_sequence_output_fine
        )
                
        return span_logits, [e_logits, pred_e_logits], total_span_loss, pred_total_type_loss, span_results, eval_types

    def evaluate_proto(self, e_mask, e_type_ids, e_type_mask, entity_types, entity_mode, is_update_type_embedding, sim_k, max_len, type_sequence_output, type_sequence_output_fine):
        total_type_loss = 0
        coarse_type_contrastive_loss, fine_type_contrastive_loss = torch.tensor(0), torch.tensor(0)
        if e_type_mask.sum() != 0:
            M = (e_type_mask[:, :, 0] != 0).max(0)[0].nonzero(as_tuple=False)[
                -1
            ].item() + 1
        else:
            M = 1
        
        e_mask = e_mask[:, :M, :max_len].type(torch.int8)  # batch_size x max_entity_num x maxlen
        e_type_ids = e_type_ids[:, :M, :]  # batch_size x max_entity_num x K
        e_type_mask = e_type_mask[:, :M, :].type(torch.int8)  # batch_size x max_entity_num x K
    
        # 细粒度label转化成粗粒度label
        coarse_e_type_ids, coarse_e_type_mask = self.covert_fine2coarse(e_type_ids, e_type_mask)
            
        ### 使用 token 构造对比学习loss
        if self.use_type_contrastive == 'token' and torch.sum(e_mask) > 1:
            coarse_type_contrastive_loss = self.calc_type_token_contrastive_loss(
                type_sequence_output,
                e_mask, 
                coarse_e_type_ids,
            )
            fine_type_contrastive_loss = self.calc_type_token_contrastive_loss(
                type_sequence_output_fine,
                e_mask, 
                e_type_ids,
            )
        
        B, M, K = e_type_ids.shape
        
        coarse_e_out = self.get_enity_hidden(
            type_sequence_output,
            e_mask,
            entity_mode,
        )
        fine_e_out = self.get_enity_hidden(
            type_sequence_output_fine,
            e_mask,
            entity_mode
        )
        
        # if self.use_classify:
        #     e_out = self.type_classify(e_out)
            
        coarse_e_out = self.ln(coarse_e_out)  # batch_size x max_entity_num x hidden_size
        fine_e_out = self.ln(fine_e_out)
        
        
        if is_update_type_embedding:
            entity_types.update_type_embedding(coarse_e_out, coarse_e_type_ids, coarse_e_type_mask)   # support set时运行这里，更新原型embedding
            entity_types.update_type_embedding(fine_e_out, e_type_ids, e_type_mask)   # support set时运行这里，更新原型embedding
            
        
        ### 用 entity 的 embedding 计算对比学习
        if self.use_type_contrastive == 'entity':
            coarse_type_contrastive_loss = self.calc_type_entity_contrastive_loss(coarse_e_out, coarse_e_type_ids, coarse_e_type_mask)
            fine_type_contrastive_loss = self.calc_type_entity_contrastive_loss(fine_e_out, e_type_ids, e_type_mask)
            
        coarse_e_out = coarse_e_out.unsqueeze(2).expand(B, M, K, -1)
        coarse_types = self.get_types_embedding(
            coarse_e_type_ids, entity_types
        )  # batch_size x max_entity_num x K x hidden_size
        
        fine_e_out = fine_e_out.unsqueeze(2).expand(B, M, K, -1)
        fine_types = self.get_types_embedding(
            e_type_ids, entity_types
        )

        sim_k = sim_k if sim_k else self.similar_k
        coarse_e_types = sim_k * (coarse_e_out * coarse_types).sum(-1) / coarse_types.shape[-1]   # batch_size x max_entity_num x K
        
        fine_e_types = sim_k * (fine_e_out * fine_types).sum(-1) / fine_types.shape[-1]   # batch_size x max_entity_num x K
        
        # fine emb 和 coarse prototypes
        new_fine_e_types = sim_k * (fine_e_out * coarse_types).sum(-1) / coarse_types.shape[-1]
        
        # coarse_e_types[coarse_e_type_mask == 0] = 0.0 # 防止计算loss的时候将正样本当做负样本
            
        coarse_e_logits = coarse_e_types
        fine_e_logits = fine_e_types
        e_logits = [coarse_e_logits, fine_e_logits] # 方便返回

        if M:
            e_type_label = torch.zeros((B, M)).to(coarse_e_types.device)   # 全0

            ##### coarse    ce
            coarse_em = coarse_e_type_mask.clone()
            coarse_em[coarse_em.sum(-1) == 0] = 1
            
            coarse_e = coarse_e_types * coarse_em   # 距离/相似度   
    
            coarse_type_loss = self.calc_loss(
                self.type_loss, coarse_e, e_type_label, coarse_e_type_mask[:, :, 0]   # e_type_mask[:, :, 0] 对应的是有效 label, 即最后一维中的 0 号位置的元素
            )
            
            ##### fine      ce
            fine_em = e_type_mask.clone()
            fine_em[fine_em.sum(-1) == 0] = 1
            
            fine_e = fine_e_types * fine_em   # 距离/相似度

            fine_type_loss = self.calc_loss(
                self.type_loss, fine_e, e_type_label, e_type_mask[:, :, 0]   # e_type_mask[:, :, 0] 对应的是有效 label, 即最后一维中的 0 号位置的元素
            )
            
            ##### fine emb 和 coarse prototypes
            new_fine_e = new_fine_e_types * coarse_em
            
            fine_type_margin_loss = self.calc_fine_margin_loss(coarse_e_type_mask, new_fine_e)
                            
        else:
            coarse_type_loss = torch.tensor(0).to(type_sequence_output.device)
            fine_type_loss = torch.tensor(0).to(type_sequence_output_fine.device)
            fine_type_margin_loss = torch.tensor(0).to(type_sequence_output_fine.device)
        
        type_contrastive_loss = coarse_type_contrastive_loss + fine_type_contrastive_loss
        
        total_type_loss = self.coarse_weight * coarse_type_loss + fine_type_loss + self.fine_margin_weight * fine_type_margin_loss + self.type_cl_weight * type_contrastive_loss
    
        return total_type_loss, e_logits
        

    
    def get_enity_hidden(
        self, hidden: torch.Tensor, e_mask: torch.Tensor, entity_mode: str
    ):
        B, M, T = e_mask.shape
        ### hidden.unsqueeze(1).expand(B, M, T, -1) 将 batch_size x seq_num x hidden_size 复制一维，变成batch_size x max_entity_num x seq_len x hidden_size
        ### e_mask.unsqueeze(-1) 展开e_mask, 变成 batch_size x max_entity_num x seq_len x 1
        ### 相乘后得到只有实体的 embedding 不为 0
        e_out = hidden.unsqueeze(1).expand(B, M, T, -1) * e_mask.unsqueeze(
            -1
        )  # batch_size x max_entity_num x seq_len x hidden_size
        
        ### sum(2) 将维度变成 batch_size x max_entity_num x hidden_size
        ### e_mask.sum(-1).unsqueeze(-1) 得到之前每个实体的长度，用于求和后取平均
        if entity_mode == "mean":
            return e_out.sum(2) / (
                e_mask.sum(-1).unsqueeze(-1) + 1e-30
            )  # batch_size x max_entity_num x hidden_size

    def get_types_embedding(self, e_type_ids: torch.Tensor, entity_types):
        return entity_types.get_types_embedding(e_type_ids)
    
    # 计算type时的对比学习loss, 用token
    def calc_type_token_contrastive_loss(self, hiddens, e_mask, e_type_ids):
        hiddens = self.type_cl_project(hiddens)
        
        B, M, _ = hiddens.shape
        # e_mask         batch_size x M x seq_len   表明句子中哪些位置有entity
        entity_labels = e_type_ids[:, :, 0]  # batch_size x M  表示句子中的实体的label类别
        
        # 获取 e_mask 中非零元素的位置索引
        # nonzero_idx = torch.nonzero(e_mask)
        # 使用 entity_labels 中对应位置的元素替换 e_mask 中的非零元素
        # e_mask[nonzero_idx[:, 0], nonzero_idx[:, 1], nonzero_idx[:, 2]] = entity_labels[nonzero_idx[:, 0], nonzero_idx[:, 1]]
        
        # 使用torch.where函数将a中的非零元素替换为b中的对应位置元素
        # 先使用torch.unsqueeze函数将b升维到与a相同的维度
        labels = torch.where(e_mask > 0, torch.unsqueeze(entity_labels, dim=2), e_mask)
        labels = torch.sum(labels, dim=1)
        labels = labels.reshape(-1)
        
        hiddens = hiddens.reshape(B*M, -1)
        
        return self.type_contrastive_loss(hiddens, labels)
    
    # 计算type时的对比学习loss, 用entity
    def calc_type_entity_contrastive_loss(self, e_out, e_type_ids, e_type_mask):
        e_out = self.type_cl_project(e_out)
        
        labels = e_type_ids[:, :, 0][e_type_mask[:, :, 0] == 1]   # 相当于取出最后一维中的 0 号位置的元素，且type_mask的值为1的   entity_num
        label_set = set(labels.detach().cpu().numpy())  # 去重后的 label 集合
        if len(label_set) == len(labels): # 没有重复的entity
            return torch.tensor(0)
        
        hiddens = e_out[e_type_mask[:, :, 0] == 1]  # 取出对应上面位置的embedding    entity_num X hidden_size 
        return self.type_contrastive_loss(hiddens, labels)
    
    # 计算span时的对比学习loss
    def calc_span_contrastive_loss(self, preds, target):
        active_loss_flag = target != -100
        active_preds = preds[active_loss_flag]
        active_target = target[active_loss_flag]
        return self.span_contrastive_loss(active_preds, active_target)
    
    def calc_fine_margin_loss(self, type_mask, preds):
        type_mask = type_mask.clone()
        preds = preds.clone()
        type_mask[:, :, 0] = 0
        
        # logger.info(f"calc------preds = {preds}")
        
        preds -= self.fine_type_margin
        type_mask[preds < 0] = 0 
        
        # logger.info(f"calc------type_mask = {type_mask}")
        preds = preds * type_mask
        
        # logger.info(f"calc------preds = {preds}")
        
        margin_loss = preds.sum() / (type_mask.sum() + 1e-10)
        
        # logger.info(f"calc------margin_loss = {margin_loss}\n")
        
        return margin_loss

    def calc_loss(self, loss_fn, preds, target, mask=None):
        target = target.reshape(-1)
        preds += 1e-10
        preds = preds.reshape(-1, preds.shape[-1])
        # logger.info(f"preds:{preds, preds.shape}")
        ce_loss = loss_fn(preds, target.long(), reduction="none")
        # logger.info(f"ce_loss:{ce_loss}")
        if mask is not None:
            mask = mask.reshape(-1)
            # logger.info(f"mask:{mask}")
            ce_loss = ce_loss * mask
            return ce_loss.sum() / (mask.sum() + 1e-10)
        return ce_loss.sum() / (target.sum() + 1e-10)
    
    def AutomaticWeightedLoss(self, mode:str, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            if loss != 0:
                weight_key = mode + str(i+1)
                loss_sum = loss_sum + 0.5 / (self.loss_weights[weight_key] ** 2) * loss + torch.log(1 + self.loss_weights[weight_key] ** 2)
            
        return loss_sum[0]
    
    def decode_span(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        types,
        mask: torch.Tensor,
        viterbi_decoder=None,
    ):
        ignore_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
        pad_token_label_id = -1
        device = target.device
        K = max([len(ii) for ii in types])
        if viterbi_decoder:
            N = target.shape[0]
            B = 16
            result = []
            for i in range((N - 1) // B + 1):
                # tmp_logits = torch.tensor(logits[i * B : (i + 1) * B]).to(target.device)
                tmp_logits = logits[i * B : (i + 1) * B]
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
                    while right < len(jj) and jj[right] == ignore_token_label_id:
                        tmp.append(-1)
                        right += 1
                    while left < len(ii):
                        tmp.append(ii[left])
                        left += 1
                        right += 1
                        while (
                            right < len(jj) and jj[right] == ignore_token_label_id
                        ):
                            tmp.append(-1)
                            right += 1
                    result.append(tmp)
        target = target.detach().cpu().numpy()
        B, T = target.shape
        if not viterbi_decoder:
            logits = torch.tensor(logits).detach().cpu().numpy()
            result = np.argmax(logits, -1)

        # if self.label_list == ["O", "B", "I"]:
        if self.num_labels == 3:
            res = []
            for ii in range(B):
                tmp, idx = [], 0
                max_pad = T - 1
                while (
                    max_pad > 0 and target[ii][max_pad - 1] == pad_token_label_id
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
                        target[ii][e + 1] == ignore_token_label_id
                        or result[ii][e + 1] in [ignore_token_label_id, 2] # 2是I 
                    ):
                        e += 1
                    tmp.append((idx, e))
                    idx = e + 1
                res.append(tmp)
        # elif self.label_list == ["O", "B", "I", "E", "S"]:
        elif self.num_labels == 5:
            res = []
            for ii in range(B):
                tmp, idx = [], 0
                max_pad = T - 1
                while (
                    max_pad > 0 and target[ii][max_pad - 1] == pad_token_label_id
                ):
                    max_pad -= 1
                while idx < max_pad:
                    if target[ii][idx] == ignore_token_label_id or (
                        result[ii][idx] not in [1, 4]
                    ):
                        idx += 1
                        continue
                    e = idx
                    while (
                        e < max_pad - 1
                        and result[ii][e] not in [3, 4]
                        and (
                            target[ii][e + 1] == ignore_token_label_id
                            or result[ii][e + 1] in [ignore_token_label_id, 2, 3]
                        )
                    ):
                        e += 1
                    if e < max_pad and result[ii][e] in [3, 4]: #检测实体右边界 遇到 E 或 S 时都算成 E
                        while (
                            e < max_pad - 1
                            and target[ii][e + 1] == ignore_token_label_id
                        ):
                            e += 1
                        tmp.append((idx, e))
                    idx = e + 1
                res.append(tmp)

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
            torch.tensor(e_mask).to(device),
            torch.tensor(e_type_ids, dtype=torch.long).to(device),
            torch.tensor(e_type_mask).to(device),
            res,
            types,
        )


class ViterbiDecoder(object):
    def __init__(
        self,
        id2label,
        transition_matrix,
        ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
    ):
        self.id2label = id2label
        self.n_labels = len(id2label)
        self.transitions = transition_matrix
        self.ignore_token_label_id = ignore_token_label_id

    def forward(self, logprobs, attention_mask, label_ids):
        # probs: batch_size x max_seq_len x n_labels
        batch_size, max_seq_len, n_labels = logprobs.size()
        attention_mask = attention_mask[:, :max_seq_len]
        label_ids = label_ids[:, :max_seq_len]

        active_tokens = (attention_mask == 1) & (
            label_ids != self.ignore_token_label_id
        )
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        label_seqs = []

        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][
                active_tokens[idx]
            ]  # seq_len(active) x n_labels

            back_pointers = []

            forward_var = logprob_i[0]  # n_labels

            for j in range(1, len(logprob_i)):  # for tag_feat in feat:
                next_label_var = forward_var + self.transitions  # n_labels x n_labels
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1)  # n_labels

                logp_j = logprob_i[j]  # n_labels
                forward_var = viterbivars_t + logp_j  # n_labels
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointers):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append(best_path)

        return label_seqs
