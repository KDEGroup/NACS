from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from GCC import QueryEncoder, GraphEncoder2
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



class InputFeaturesTrip(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTrip, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids

# 中序遍历， 顺便给每个ast节点加上了id
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

def convert_examples_to_features(js, tokenizer, args, ast_vocab, query_vocab):
    # label
    label = js['label']

    # code
    code = js['code']
    if args.code_type == 'code_tokens':
        code = js['code_tokens']

    code_tokens = tokenizer.tokenize(code)[:args.max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    # ast_vocab = pickle.load(open(args.ast_vocab_path, 'rb'))

    tree = ast.parse(code).body[0]
    # 标记下id和parent
    get_ast_seq_level(tree)
    ast_graph = get_ast_graph(tree, ast_vocab)
    ast_graph = _add_undirected_graph_positional_embedding(ast_graph, 32)

    nl = js['doc']

    nl_tokens = tokenizer.tokenize(nl)[:args.max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    query = js['doc']
    query = query.replace('\n', ' ')  # 去掉回车
    query = re.sub(
        r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', query)  # 去掉括号内容
    query = query.replace('_', ' ').strip()
    query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
    query = split_camel(query)  # 拆驼峰命名
    query = tokenizer1(query_vocab, query)
    query = pad_seq(query, 100, 0)

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, query, ast_graph, label, js['idx'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        # file：json文件，每一个dict中包括：idx, query, doc, code(或者叫function_tokens，list形式), docstring_tokens(list形式)
        self.examples = []
        self.data=[]
        ast_vocab = pickle.load(open(args.ast_vocab_path, 'rb'))
        query_vocab = pickle.load(open(args.query_vocab_path, 'rb'))
        count = 0
        if os.path.exists(os.path.join(args.data_dir, 'train_data.pickle')):
            self.examples = pickle.load(open(os.path.join(args.data_dir, 'train_data.pickle'), 'rb'))
        else:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            for js in self.data:
                if js['label'] == 1:
                    self.examples.append(convert_examples_to_features(js, tokenizer, args, ast_vocab, query_vocab))
            
            pickle.dump(self.examples, open(os.path.join(args.data_dir, 'train_data.pickle'), 'wb'))
        print()
        # 如果是training集，就print前3个
        # if 'train' in file_path:
        #     for idx, example in enumerate(self.examples[:3]):
        #         logger.info("*** Example ***")
        #         logger.info("idx: {}".format(idx))
        #         logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
        #         logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
        #         logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
        #         logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return self.examples[i].code_ids, \
               self.examples[i].nl_ids,\
               self.examples[i].label,\
               self.examples[i].query,\
               self.examples[i].ast_graph
               # torch.tensor(i)

class RetievalDataset(Dataset):
    def __init__(self, tokenizer, args, data_path=None):
        self.codes = []
        self.data = []
        self.examples = []  # codebase用code和code当成6k pair; testdata用query和code当成pair来算; examples前0-6267，code为6267-6767
        code_file = args.retrieval_code_base
        data_file = data_path
        ast_vocab = pickle.load(open(args.ast_vocab_path, 'rb'))
        query_vocab = pickle.load(open(args.query_vocab_path, 'rb'))
        if os.path.exists(os.path.join(args.data_dir, args.saved_test_data)):
            self.examples = pickle.load(open(os.path.join(args.data_dir, args.saved_test_data), 'rb'))
        else:
            with open(code_file, 'r') as f:
                self.codes = json.loads(f.read())
            for code, code_id in self.codes.items():
                js = {'code': code, 'doc': code, 'label': 1, 'idx': code_id}
                self.examples.append(convert_examples_to_features(js, tokenizer, args, ast_vocab, query_vocab))
            with open(data_file, 'r') as f:
                self.data = json.load(f)
            for js in self.data:
                new_js = {'code': js['code'], 'doc': js['doc'], 'label': js['label'], 'idx': js['retrieval_idx']}
                self.examples.append(convert_examples_to_features(new_js, tokenizer, args, ast_vocab, query_vocab))
            pickle.dump(self.examples, open(os.path.join(args.data_dir, args.saved_test_data), 'wb'))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return self.examples[i].code_ids, \
               self.examples[i].nl_ids,\
               self.examples[i].label,\
               self.examples[i].query,\
               self.examples[i].ast_graph


def set_seed(seed=45):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def batcher():
    def batcher_dev(batch):
        code_ids, nl_ids, labels, query, ast_graphs = zip(*batch)
        #batch的效果是将 n 张小图打包在一起的操作可以看成是生成一张含 n 个不相连小图的大图。
        ast_graphs = dgl.batch(ast_graphs)
        return code_ids, nl_ids, labels, query, ast_graphs

    return batcher_dev

def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True, collate_fn=batcher())

    args.save_steps = len(train_dataloader) if args.save_steps<=0 else args.save_steps
    args.warmup_steps = len(train_dataloader) if args.warmup_steps<=0 else args.warmup_steps
    args.logging_steps = len(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps)
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    # optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last))
    # if os.path.exists(optimizer_last):
    #     optimizer.load_state_dict(torch.load(optimizer_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0

    # best_results = {"acc": -1.0, "precision": -1.0, "recall": -1.0, "f1": -1.0, "acc_and_f1": -1.0}
    best_results = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "acc_and_f1": 0.0, "mrr": 0.0}
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    # logger.info(model)
    mrr = evaluate(args, model, tokenizer, eval_when_training=True)
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(enumerate(train_dataloader))
        tr_num=0
        tr_loss=0
        for step, batch in bar:
            if torch.sum(torch.tensor(batch[2])) <= 1:
                continue
            code_inputs = torch.tensor(batch[0]).to(args.device)
            nl_inputs = torch.tensor(batch[1]).to(args.device)
            # ds_inputs = batch[2].to(args.device)
            labels = torch.tensor(batch[2]).to(args.device)

            query = torch.tensor(batch[3]).to(args.device)
            ast_graph = batch[4].to(args.device)


            model.train()
            loss = model(code_inputs, nl_inputs, query, ast_graph, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(tr_loss/tr_num, 5)
            bar.set_description("epoch {} step {} loss {}".format(idx, step+1, avg_loss))

            # args.gradient_accumulation_steps 值是1，就是说每步都会调整optimizer和scheduler
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                # avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     logging_loss = tr_loss
                #     tr_nb = global_step
                # 每个epoch都会eval一次，并且保存最好的mrr模型到checkpoint-best-mrr这个文件夹里
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        mrr = evaluate(args, model, tokenizer, eval_when_training=True)
                        logger.info(" Mrr = %s", round(mrr, 4))
                        # Save model checkpoint
                        if mrr >= best_results['mrr']:
                            best_results['mrr'] = mrr

                            # save
                            checkpoint_prefix = 'checkpoint-best-mrr'
                            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model

                            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                            logger.info("Saving model checkpoint to %s", output_dir)

                    # 每个epoch都会保存最新的模型到checkpoint-last这个文件夹里
                    if args.local_rank == -1:

                        # save
                        checkpoint_prefix = 'checkpoint-all'
                        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model_{}.bin'.format(idx)))

                        checkpoint_prefix = 'checkpoint-last'
                        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                        tokenizer.save_pretrained(output_dir)

                        idx_file = os.path.join(output_dir, 'idx_file.txt')
                        with open(idx_file, 'w', encoding='utf-8') as idxf:
                            idxf.write(str(args.start_epoch + idx) + '\n')

                        step_file = os.path.join(output_dir, 'step_file.txt')
                        with open(step_file, 'w', encoding='utf-8') as stepf:
                            stepf.write(str(global_step) + '\n')



eval_dataset=None
def evaluate(args, model, tokenizer,eval_when_training=False):
    """ dev集上时，应该使用一样的计算方法 """
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
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

def main():

    gcc_ratio = 0.0001
    batch_size = 16
    args = {
        "nce_gcc": 0.07,
        "nce_cb":0.07,
        "gcc_ratio":gcc_ratio,
        "output_dir": 'search/model/search_gcc_add_qc', 
        "ast_encode_path": "ast_pretrain/saved_models/GCC_CSN_all_add_cp/pretrain.pth",
        'saved_test_data': "test_data.pickle",
        "model_type": "roberta", 
        "augment": True, 
        "do_train": False, 
        "do_eval": True, 
        "eval_all_checkpoints": True, 
        "data_dir": "data/search/", 
        "train_data_file": f"cosqa-retrieval-train-19604.json", 
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
        "encoder_name_or_path": "microsoft/codebert-base",
        "config_name": "",
        "tokenizer_name": "",
        "cache_dir": "codesearch/cache",
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
        "ast_vocab_path":"data/pretrain/ast_vocab.pickle",
        "query_vocab_path":"data/pretrain/query_vocab.pickle",
    }
    args = argparse.Namespace(**args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    # checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    # if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
    #     # args.encoder_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
    #     args.config_name = os.path.join(checkpoint_last, 'config.json')
    #     idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
    #     with open(idx_file, encoding='utf-8') as idxf:
    #         args.start_epoch = int(idxf.readlines()[0].strip()) + 1

    #     step_file = os.path.join(checkpoint_last, 'step_file.txt')
    #     if os.path.exists(step_file):
    #         with open(step_file, encoding='utf-8') as stepf:
    #             args.start_step = int(stepf.readlines()[0].strip())

    #     logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.encoder_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    # if args.max_seq_length <= 0:
    #     args.max_seq_length = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    # args.max_seq_length = min(args.max_seq_length, tokenizer.max_len_single_sentence)
    # if args.encoder_name_or_path:

    # 是否真的有把预训练的模型加载进来，
    bert_model = model_class.from_pretrained(args.encoder_name_or_path,
                                        from_tf=bool('.ckpt' in args.encoder_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    # else:
    #     model = model_class(config)

    ast_encoder = GraphEncoder2()
    query_encoder = QueryEncoder()
    
    if args.ast_encode_path:
        ast_encoder.load_state_dict(torch.load(args.ast_encode_path, map_location={'cuda:3':'cuda:0'})['model'], strict=False)
        # ast_encoder.load_state_dict(torch.load(args.ast_encode_path, map_location={'cuda:1':'cuda:0'})['model'], strict=False)
    
    if args.ast_encode_path:
        query_encoder.load_state_dict(torch.load(args.ast_encode_path, map_location={'cuda:3':'cuda:0'})['model_query'], strict=False)
        # query_encoder.load_state_dict(torch.load(args.ast_encode_path, map_location={'cuda:1':'cuda:0'})['model_query'], strict=False)
    # model = ModelContraASTGraph(bert_model, config, tokenizer, args, ast_encoder)

    model = ModelContraASTGraphQuerySplit(bert_model, config, tokenizer, args, ast_encoder, query_encoder)

    # if args.checkpoint_path:
    #     model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin')))
    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # if args.local_rank not in [-1, 0]:
        #     torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        train_data_path = os.path.join(args.data_dir, args.train_data_file)
        train_dataset = TextDataset(tokenizer, args, train_data_path)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-mrr'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        # model.load_state_dict(torch.load('model/search_codebert_ast1/checkpoint-best-mrr/training_18.bin'))
        tokenizer = tokenizer.from_pretrained(output_dir)
        model.to(args.device)
        mrr = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        logger.info("  Eval MRR = %s", str(mrr))
        logger.info("Eval Model From: {}".format(os.path.join(output_dir, 'pytorch_model.bin')))
        logger.info("***** Eval results *****")

        return results


if __name__ == "__main__":
    main()


