from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import random

import torch as th
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

from tqdm import tqdm, trange
import multiprocessing
from models import ModelContraASTGraphQuerySplit
from utils import acc_and_f1
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from GCC import  QueryEncoder, GraphEncoder2
import argparse
import ast
from vocab import WordVocab
import dgl
from scipy.sparse import linalg
import sklearn.preprocessing as preprocessing
import scipy.sparse as sparse
import torch.nn.functional as F

import warnings
import re

from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
language = 'python'
LANGUAGE = Language('code_search/parser/my-languages.so', language)
parser = Parser()
parser.set_language(LANGUAGE)



warnings.filterwarnings("ignore")



logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, query, ast_graph, label, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids        # code tokenize后的ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids            # nl tokenize后的ids
        self.label = label
        self.idx = idx
        self.ast_graph = ast_graph
        self.query = query

class RetievalDataset(Dataset):
    def __init__(self, tokenizer, args, data_path=None):
        self.codes = []
        self.data = []
        self.examples = []  # codebase用code和code当成6k pair; testdata用query和code当成pair来算; examples前0-6267，code为6267-6767
        
        if os.path.exists(os.path.join(args.data_dir, args.saved_test_data)):
            self.examples = pickle.load(open(os.path.join(args.data_dir, args.saved_test_data), 'rb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return self.examples[i].code_ids, \
               self.examples[i].nl_ids,\
               self.examples[i].label,\
               self.examples[i].query,\
               self.examples[i].ast_graph

def batcher():
    def batcher_dev(batch):
        code_ids, nl_ids, labels, query, ast_graphs = zip(*batch)
        #batch的效果是将 n 张小图打包在一起的操作可以看成是生成一张含 n 个不相连小图的大图。
        ast_graphs = dgl.batch(ast_graphs)
        return code_ids, nl_ids, labels, query, ast_graphs

    return batcher_dev

def evaluate(args, model, tokenizer,eval_when_training=False):
    """ dev集上时，应该使用一样的计算方法 """
    eval_output_dir = args.output_dir
    args.retrieval_code_base = os.path.join(args.data_dir, args.retrieval_code_base)
    eval_data_path = os.path.join(args.data_dir, args.eval_data_file)
    eval_dataset = RetievalDataset(tokenizer, args, eval_data_path)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_retrieval_batch_size * max(1, args.n_gpu)
    
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, collate_fn=batcher(), pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_first_vec = []
    all_second_vec = []
    all_third_vec = []
    all_fourth_vec = []
    for batch in eval_dataloader:

        code_inputs = torch.tensor(batch[0]).to(args.device)
        nl_inputs = torch.tensor(batch[1]).to(args.device)
        # ds_inputs = batch[2].to(args.device)
        labels = torch.tensor(batch[2]).to(args.device)

        query = torch.tensor(batch[3]).to(args.device)
        ast_graph = batch[4].to(args.device)

        with torch.no_grad():
            code_vec, nl_vec, ast_vec, query_vec = model(code_inputs, nl_inputs, query, ast_graph, labels, return_vec=True)
            all_first_vec.append(code_vec.cpu())
            all_second_vec.append(nl_vec.cpu())
            all_third_vec.append(ast_vec.cpu())
            all_fourth_vec.append(query_vec.cpu())

        nb_eval_steps += 1
    code_vectors = torch.cat(all_first_vec, 0).squeeze()[:6267, :]
    # print(code_vectors.size())
    nl_vectors = torch.cat(all_second_vec, 0).squeeze()[6267:, :]
    ast_vectors = torch.cat(all_third_vec, 0).squeeze()[:6267, :]
    # print(code_vectors.size())
    query_vectors = torch.cat(all_fourth_vec, 0).squeeze()[6267:, :]

    # print(nl_vectors.size())
    assert(code_vectors.size(0)==6267)
    assert(nl_vectors.size(0)==500)

    scores = torch.matmul(nl_vectors, code_vectors.t()) + args.gcc_ratio * torch.matmul(query_vectors, ast_vectors.t())
    # scores = torch.matmul(nl_vectors, code_vectors.t())
    # repeat_nl_vec = query_vectors.unsqueeze(1).to(args.device)
    # scores = []
    # with torch.no_grad():
    #     for idx in trange(6267):
    #         repeat_code_vec = code_vectors[idx, :].unsqueeze(0).repeat([500, 1, 1]).to(args.device)
    #         logits = model.module.mlp(torch.cat((repeat_nl_vec, repeat_code_vec, repeat_nl_vec - repeat_code_vec, repeat_nl_vec * repeat_code_vec),2)).squeeze(2)
    #         # logits = model.module.mlp(torch.cat((repeat_nl_vec, repeat_code_vec),2)).squeeze(2)
            
    #         scores.append(logits.cpu())

    # scores = torch.cat(scores, 1)  
    results = []
    mrr = 0
    for idx in range(len(scores)):
        rank = torch.argsort(-scores[idx]).tolist()
        # logger.info(rank[:10])
        example = eval_dataset.examples[idx+6267]
        # logger.info(example.idx)
        item = {}
        item['ans'] = example.idx
        item['rank'] = rank
        item['rr'] = 1 / (rank.index(example.idx)+1)
        mrr += item['rr']
        results.append(item)
    mrr = mrr / len(scores)
    logger.info("  Final test MRR {}".format(mrr))
    return mrr



def split_camel(camel_str, test=''):
    try:
        split_str = re.sub(
            r'(?<=[a-z]|[0-9])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+',
            '_',
            camel_str)
    except TypeError:
        return ['']
    try:
        if split_str[0] == '_':
            return [camel_str]
    except IndexError:
        return []
    return [i for i in split_str.lower().split('_') if i != '']

def tokenizer1(vocab, seq):
    for i in range(len(seq)):
        seq[i] = vocab.stoi.get(seq[i], vocab.unk_index)
    return seq

def pad_seq(seq, max_seq_len, pad_id):
    if len(seq) < max_seq_len:
        new_seq = seq + [pad_id] * (max_seq_len - len(seq))
    else:
        new_seq = seq[:max_seq_len]
    return new_seq

def get_ast_seq_level(tree):
    astnodes = []
    nodes = []
    current_idx = 0
    for node in ast.walk(tree):
        # 因为load，store这种可能会重复出现，所以加个判断，重复出现就不重新加id了
        if node not in nodes:
            nodes.append(node)
            node.idx = current_idx
            current_idx += 1
        # 加上父节点
        if not next(ast.iter_child_nodes(node), None) is None:
            for child in ast.iter_child_nodes(node):
                child.parent = node

        astnodes.append(type(node).__name__)
    return astnodes

def get_ast_graph(tree, ast_vocab):
    u = []
    v = []
    node_type = []
    astnodes = []
    for node in ast.walk(tree):
        if node not in astnodes:
            astnodes.append(node)
            node_type.append(ast_vocab.stoi.get(
                type(node).__name__, ast_vocab.unk_index))
        if not next(ast.iter_child_nodes(node), None) is None:
            for child in ast.iter_child_nodes(node):
                u.append(int(node.idx))
                v.append(int(child.idx))
    node_type = th.tensor(node_type)
    u = th.tensor(u)
    v = th.tensor(v)
    graph = dgl.graph((u, v))
    # 无向图
    graph = dgl.to_bidirected(graph)
    graph.ndata['node_type'] = node_type
    return graph

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64") 
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0) # 特征值 特征向量
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float) # 邻接矩阵
    # 每个点的入度，做成一个对角矩阵
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    # 拉普拉斯矩阵就是这么乘出来的
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    g.ndata["seed"] = torch.zeros(g.number_of_nodes(), dtype=torch.long)
    g.ndata["seed"][0] = 1
    return g

if __name__ == "__main__":
    # 基础配置
    gcc_ratio = 0.0001
    batch_size = 48
    args = {
        "nce_gcc": 1,
        "nce_cb":1,
        "gcc_ratio":gcc_ratio,
        "output_dir": '/data/jdfeng/CoCLR/model_10_2/search_gcc_add_qc_' + str(gcc_ratio) + '_0.07_0.07_70_switch_VN_lr6e-5', 
        # "output_dir": 'good_model/search_gcc_add_qc_0.0001_0.07_0.07_70_switch_lr6e-5', 
        "ast_encode_path": "/data/jdfeng/CoCLR/ast_pretrain/saved_models/GCC_CSN_all_add_cp/ckpt_epoch_70.pth",
        'saved_test_data': "test_data_old.pickle",
        "model_type": "roberta", 
        "augment": True, 
        "do_train": False, 
        "do_eval": True, 
        "eval_all_checkpoints": True, 
        "data_dir": "/data/jdfeng/CoCLR/data/search/", 
        "train_data_file": f"cosqa-retrieval-train-19604-qra-switch-VN-37644.json", 
        "eval_data_file": "cosqa-retrieval-test-500.json", 
        "retrieval_code_base": "code_idx_map.txt", 
        "code_type": "code", 
        "max_seq_length": 200, 
        "per_gpu_train_batch_size": batch_size, 
        "per_gpu_retrieval_batch_size": 67, 
        "learning_rate": 6e-5, 
        "num_train_epochs": 20, 
        "gradient_accumulation_steps": 1, 
        "evaluate_during_training": True, 
        "checkpoint_path": "/data/jdfeng/CoCLR/model/codesearchnet/checkpoint-last", 
        "encoder_name_or_path": "microsoft/codebert-base",
        "config_name": "",
        "tokenizer_name": "",
        "cache_dir": "/data/jdfeng/CoCLR/code_search/cache",
        "do_lower_case": False,
        "mrr_rank": 100,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 0,
        "logging_steps": 50,
        "save_steps": 0,
        "save_total_limit": None,
        "seed": 45,
        "no_cuda": False,
        "fp16": False,
        "fp16_opt_level": "01",
        "local_rank": -1,
        "server_ip": "",
        "server_port": "",
        "pred_model_dir": None,
        "test_predictions_output": None,
        "retrieval_predictions_output": None,
        "do_retrieval": False,
        "ast_vocab_path":"/data/jdfeng/CoCLR/data/pretrain/ast_vocab.pickle",
        "query_vocab_path":"/data/jdfeng/CoCLR/data/pretrain/query_vocab.pickle",
    }
    args = argparse.Namespace(**args)

    # gpu配置
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device


    # model生成和加载，路径可能需要变一下
    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.encoder_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    bert_model = model_class.from_pretrained(args.encoder_name_or_path,
                                        from_tf=bool('.ckpt' in args.encoder_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    ast_encoder = GraphEncoder2()
    query_encoder = QueryEncoder()

    model = ModelContraASTGraphQuerySplit(bert_model, config, tokenizer, args, ast_encoder, query_encoder)

    model.load_state_dict(torch.load('good_model/search_gcc_add_qc_0.0001_0.07_0.07_70_switch_lr6e-5/checkpoint-best-mrr/pytorch_model.bin'))
    # model.load_state_dict(torch.load('model/search_codebert_ast1/checkpoint-best-mrr/training_18.bin'))
    tokenizer = tokenizer.from_pretrained('good_model/search_gcc_add_qc_0.0001_0.07_0.07_70_switch_lr6e-5/checkpoint-best-mrr/')
    model.to(args.device)



    # 数据的转换
    ast_vocab = pickle.load(open(args.ast_vocab_path, 'rb'))
    query_vocab = pickle.load(open(args.query_vocab_path, 'rb'))

    # nl和query
    nl = 'hello world'
    query = 'hello world'

    code1 = '''
def test1():
    print()
'''

    code2 = '''
def test2():
    a = 1
    b = 2
    c = a + b
    return c
'''

    # 对数据进行处理
    nl_tokens = tokenizer.tokenize(nl)[:args.max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length


    query = query.replace('\n', ' ')  # 去掉回车
    query = re.sub(
        r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', query)  # 去掉括号内容
    query = query.replace('_', ' ').strip()
    query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
    query = split_camel(query)  # 拆驼峰命名
    query = tokenizer1(query_vocab, query)
    query = pad_seq(query, 100, 0)


    # 对代码处理
    code1 = remove_comments_and_docstrings(code1.expandtabs(tabsize=4), 'python')
    tree = parser.parse(bytes(code1, 'utf8'))
    tokens_index = tree_to_token_index(tree.root_node)
    code_split=code1.split('\n')
    code_tokens = []
    for x in tokens_index:
        code_tokens += index_to_code_token(x, code_split)

    code_tokens = ' '.join(code_tokens)
    code_tokens = tokenizer.tokenize(code_tokens)[:args.max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    tree = ast.parse(code1).body[0]
    # 标记下id和parent
    get_ast_seq_level(tree)
    ast_graph = get_ast_graph(tree, ast_vocab)
    ast_graph = _add_undirected_graph_positional_embedding(ast_graph, 32)

    code_inputs = torch.tensor([code_ids]).to(args.device)
    nl_inputs = torch.tensor([nl_ids]).to(args.device)
    query = torch.tensor([query]).to(args.device)
    ast_graph = ast_graph.to(args.device)
    labels = torch.tensor([1]).to(args.device)
    
    model.eval()

    # 返回向量
    code_vec, nl_vec, ast_vec, query_vec = model(code_inputs, nl_inputs, query, ast_graph, labels, return_vec=True)
    
    code_vec = model.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
    nl_vec = model.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
    query_vec = model.query_encoder(query)
    ast_vec = model.ast_encoder(ast_graph)

    # 计算相似度
    scores = torch.matmul(nl_vec, code_vec.t()) + args.gcc_ratio * torch.matmul(query_vec, ast_vec.t())

    # 你们做的时候可以提前吧代码库所有代码的向量表示提前计算好，那么每次有一个新的查询来的时候，
    # 你们就只需要给这个查询进行一个编码，不需要对所有代码进行一个编码了



    print()
    # mrr = evaluate(args, model, tokenizer)
    # logger.info("***** Eval results *****")
    # logger.info("  Eval MRR = %s", str(mrr))
    # logger.info("***** Eval results *****")