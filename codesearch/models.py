import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from transformers.modeling_utils import PreTrainedModel
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 图结构部分和代码部分分开
class ModelContraASTGraphQuerySplit(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args, ast_encoder=None, query_encoder=None):
        super(ModelContraASTGraphQuerySplit, self).__init__(config)
        self.nce_gcc = args.nce_gcc
        self.nce_cb = args.nce_cb
        self.gcc_ratio = args.gcc_ratio
        self.encoder = encoder
        self.ast_encoder = ast_encoder
        self.query_encoder = query_encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())

        # self.query_linear = nn.Linear(768, 768)
        self.code_linear = nn.Linear(768 + 128, 768)

        self.loss_func = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, query, ast_graph, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]

        code_vec = F.normalize(code_vec, p=2, dim=-1, eps=1e-5)
        nl_vec = F.normalize(nl_vec, p=2, dim=-1, eps=1e-5)
        sims1 = torch.matmul(nl_vec, code_vec.t()) / self.nce_cb
        sims2 = torch.matmul(code_vec, nl_vec.t()) / self.nce_cb

        # 直接两两拼接计算相似度
        ast_vec = self.ast_encoder(ast_graph)
        query_vec = self.query_encoder(query)

        sims3 = torch.matmul(query_vec, ast_vec.t()) / self.nce_gcc
        sims4 = torch.matmul(ast_vec, query_vec.t()) / self.nce_gcc


        sims1 = sims1 + self.gcc_ratio * sims3
        sims2 = sims2 + self.gcc_ratio * sims4

        sims1 = sims1[labels == 1]
        sims2 = sims2[labels == 1]
        sims3 = sims3[labels == 1]
        sims4 = sims4[labels == 1]

        pos_size = sims1.shape[0]

        label = torch.nonzero(labels==1).squeeze()

        loss = (self.loss_func(sims1, label) + self.loss_func(sims2, label)) / pos_size

        if return_vec:
            return code_vec, nl_vec, ast_vec, query_vec
        else:
            return loss

