import json
from tqdm import tqdm
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from vocab import WordVocab
import re
import pickle
import os
import torch as th
import dgl
from dgl.data.utils import save_graphs, load_graphs
import datetime
import random
from queue import Queue


JAVA_LANGUAGE = Language(tsjava.language(), "java")
parser = Parser()
parser.set_language(JAVA_LANGUAGE)


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


def get_nodes_level(code):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    # 遍历树，收集所有节点的类型
    tokens = []

    # Traverse the tree-sitter syntax tree and collect token names
    def traverse_node(node):
        node_type = node.type
        tokens.append(node_type)

        # Recursively process child nodes
        for child in node.children:
            traverse_node(child)

    # Start traversal from the root
    traverse_node(root_node)

    return tokens


"""
def get_ast_seq_level(tree):
    node_idx_map = {}  # 用于存储节点和对应的索引
    current_idx = 0

    # Helper function to recursively walk through the tree-sitter nodes
    def walk_tree_sitter(node, parent=None):
        nonlocal current_idx

        # 如果该节点没有被记录，给它分配一个索引
        if node not in node_idx_map:
            node_idx_map[node] = current_idx
            current_idx += 1

        # 遍历子节点
        for child in node.children:
            walk_tree_sitter(child, node)  # 递归遍历子节点

    # 从根节点开始遍历
    root_node = tree.root_node
    walk_tree_sitter(root_node)
"""


def get_vars(tree):
    root_node = tree.root_node
    identifiers = []

    def traverse_node(node, parent=None):
        if node.type == 'identifier' and node.parent.type not in {'class_declaration', 'method_declaration', "argument_list"}:
            if not((node.parent and node.parent.type == 'method_invocation' and node.parent.child_by_field_name('name') == node) or (node.parent and node.parent.type == 'field_access' and node.parent.child_by_field_name('object') != node)):
                identifiers.append(node.text.decode('utf-8'))

        # Recursively process child nodes
        for child in node.children:
            traverse_node(child, node)

    # Start traversal from the root
    traverse_node(root_node)

    return list(set(identifiers))


def tokenizer(vocab, seqs, output=None):
    for seq in seqs:
        for i in range(len(seq)):
            # 如果是列表的话，就要再来一个for循环
            if isinstance(seq[i], list):
                for j in range(len(seq[i])):
                    seq[i][j] = vocab.stoi.get(seq[i][j], vocab.unk_index)
            else:
                seq[i] = vocab.stoi.get(seq[i], vocab.unk_index)
    if output:
        f1 = open(output, 'wb')
        pickle.dump(seqs, f1)
    return seqs


def get_ast_graph(tree, ast_vocab):
    u = []
    v = []
    node_type = []
    node_index = {}  # 用于存储节点的索引
    current_idx = 0  # 用于分配节点的索引

    # 递归遍历 Tree-sitter 树
    def traverse_node(node):
        nonlocal current_idx
        if node not in node_index:
            # 记录当前节点索引
            node_index[node] = current_idx
            current_idx += 1
            # 获取节点类型并加入 node_type 列表
            node_type.append(ast_vocab.stoi.get(node.type, ast_vocab.unk_index))

        # 遍历子节点并生成边
        for child in node.children:
            u.append(node_index[node])
            v.append(current_idx)
            traverse_node(child)

    # 从根节点开始遍历
    traverse_node(tree.root_node)

    # 转换为 tensor 格式
    node_type = th.tensor(node_type)
    u = th.tensor(u)
    v = th.tensor(v)

    # 构建图
    graph = dgl.graph((u, v))
    graph = dgl.to_bidirected(graph)  # 无向图

    # 设置图节点数据
    graph.ndata['node_type'] = node_type
    graph.ndata['degree'] = graph.in_degrees()

    return graph

# 获取对比学习的新图，随机删掉一个节点（一颗子树）
def get_ast_graph_new(tree, ast_vocab, del_id):
    node_idx_map = {}
    current_idx = 0

    def walk_tree_sitter(node, parent=None):
        nonlocal current_idx
        if node not in node_idx_map:
            node_idx_map[node] = current_idx
            current_idx += 1
        for child in node.children:
            walk_tree_sitter(child, node)

    walk_tree_sitter(tree.root_node)

    u = []
    v = []
    node_type = []
    astnodes = []
    current_idx = 0
    node_queue = Queue()
    node_queue.put(tree.root_node)
    new_node_idx_map = {}

    # 第一遍遍历：给所有节点分配索引，并跳过 del_id 节点
    while not node_queue.empty():
        cur_node = node_queue.get()
        if cur_node not in astnodes:
            astnodes.append(cur_node)
            new_node_idx_map[cur_node] = current_idx
            current_idx += 1

        # 遍历子节点，跳过 del_id 对应的节点
        for child in cur_node.children:
            if node_idx_map[child] != del_id:  # 检查节点的唯一 id
                node_queue.put(child)

    # 第二遍遍历：重新建立图的边关系，跳过 del_id 节点
    astnodes = []
    node_queue = Queue()
    node_queue.put(tree.root_node)
    while not node_queue.empty():
        cur_node = node_queue.get()
        if cur_node not in astnodes:
            astnodes.append(cur_node)
            # 记录节点类型
            node_type.append(ast_vocab.stoi.get(cur_node.type, ast_vocab.unk_index))

        # 遍历子节点并构建边
        for child in cur_node.children:
            if node_idx_map[child] != del_id:
                node_queue.put(child)
                u.append(new_node_idx_map[cur_node])
                v.append(new_node_idx_map[child])

    # 转换为 tensor 格式
    node_type = th.tensor(node_type)
    u = th.tensor(u)
    v = th.tensor(v)

    # 构建图
    graph = dgl.graph((u, v))
    graph = dgl.to_bidirected(graph)  # 无向图

    # 设置图节点数据
    graph.ndata['node_type'] = node_type
    graph.ndata['degree'] = graph.in_degrees()

    return graph


def get_token_vocab(data_path):
    train_data_path = os.path.join(data_path, 'csn_train.json')
    query_vocab_path = os.path.join(data_path, 'query_vocab.pickle')
    ast_vocab_path = os.path.join(data_path, 'ast_vocab.pickle')
    if os.path.exists(query_vocab_path) and os.path.exists(ast_vocab_path):
        query_vocab = pickle.load(open(query_vocab_path, 'rb'))
        ast_vocab = pickle.load(open(ast_vocab_path, 'rb'))
    else:
        vocab_query = []
        vocab_ast = []
        f = open(train_data_path)
        js = json.load(f)

        for line in tqdm(js):
            code = line['code']
            # code_tokens = line['code_tokens']
            query = line['doc']

            query = query.replace('\n', ' ')  # 去掉回车
            query = re.sub(
                r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', query)  # 去掉括号内容
            query = query.replace('_', ' ').strip()
            query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
            query_tokens = split_camel(query)  # 拆驼峰命名

            ast_nodes = get_nodes_level(code)

            if not query_tokens or not ast_nodes:
                print(1234)

            vocab_query += query_tokens
            vocab_ast += ast_nodes

        query_vocab = WordVocab(vocab_query, max_size=50000,
                                min_freq=1)
        query_vocab.save_vocab(query_vocab_path)

        ast_vocab = WordVocab(vocab_ast, max_size=10000,
                              min_freq=1)
        ast_vocab.save_vocab(ast_vocab_path)

    return query_vocab, ast_vocab


# 先序遍历
# node是当前节点，paths, tmp, astnodes是记录路径，当前路径，和当前遍历
def get_ast_seq(node, astnodes, paths, tmp):
    # 记录当前节点类型
    astnodes.append(node.type)
    tmp.append(node.type)

    # 如果是叶子节点，记录路径
    if len(node.children) == 0:
        paths.append(tmp[:])
    else:
        # 遍历子节点
        for child in node.children:
            get_ast_seq(child, astnodes, paths, tmp)

    # 弹出当前节点
    tmp.pop()


def get_train_data(query_vocab, ast_vocab, data_path):
    train_data_path = os.path.join(data_path, 'csn_train.json')
    train_query_path = os.path.join(data_path, 'query_dataset_train.pickle')
    var_vocab_path = os.path.join(data_path, 'var_vocab.pickle')
    train_ast_graph_path = os.path.join(data_path, 'graph_data_train_nodetype.pickle')
    codes_path = os.path.join(data_path, 'train_codes.pickle')
    if os.path.exists(train_query_path) and os.path.exists(var_vocab_path) and os.path.exists(train_ast_graph_path) and os.path.exists(codes_path):
        query_seq_list = pickle.load(open(train_query_path, 'rb'))
        var_vocab = pickle.load(open(var_vocab_path, 'rb'))
        ast_graph_list, _ = load_graphs(train_ast_graph_path)
        codes = pickle.load(open(codes_path, 'rb'))
    else:
        f = open(train_data_path)
        lines = json.load(f)
        query_seq_list = []
        code_seq_list = []
        codes = []
        var_vocab = []
        ast_seq_list = []
        ast_seq_level_list = []
        ast_graph_list = []
        # 从根节点到所有叶子节点的路径
        ast_path_list = []

        for js in tqdm(lines):
            query = js['doc']
            query = query.replace('\n', ' ')  # 去掉回车
            query = re.sub(
                r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', query)  # 去掉括号内容
            query = query.replace('_', ' ').strip()
            query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
            query_seq_list.append(split_camel(query))  # 拆驼峰命名

            code = js['code']
            codes.append(code)
            tree = parser.parse(bytes(code, "utf8"))
            
            # 标记下id和parent
            #get_ast_seq_level(tree)

            var_vocab += get_vars(tree)

            ast_graph_list.append(get_ast_graph(tree, ast_vocab))

        query_seq_list = tokenizer(
            query_vocab, query_seq_list, train_query_path)

        save_graphs(train_ast_graph_path, ast_graph_list)

        pickle.dump(list(set(var_vocab)), open(var_vocab_path, 'wb'))
        pickle.dump(codes, open(codes_path, 'wb'))
        
    return query_seq_list, codes, var_vocab, ast_graph_list



def get_train_data_new(query_vocab, ast_vocab, data_path):
    train_data_path = os.path.join(data_path, 'csn_train.json')
    train_query_path = os.path.join(data_path, 'query_dataset_train.pickle')
    var_vocab_path = os.path.join(data_path, 'var_vocab.pickle')
    train_ast_graph_path = os.path.join(data_path, 'graph_data_train_nodetype.pickle')
    codes_path = os.path.join(data_path, 'train_codes.pickle')
    ast_path = os.path.join(data_path, 'ast_path.pickle')
    if os.path.exists(train_query_path) and os.path.exists(var_vocab_path) and os.path.exists(train_ast_graph_path) and os.path.exists(codes_path) and os.path.exists(ast_path):
        query_seq_list = pickle.load(open(train_query_path, 'rb'))
        var_vocab = pickle.load(open(var_vocab_path, 'rb'))
        ast_graph_list, _ = load_graphs(train_ast_graph_path)
        codes = pickle.load(open(codes_path, 'rb'))
        ast_path_list = pickle.load(open(ast_path, 'rb'))
    else:
        f = open(train_data_path)
        lines = json.load(f)
        query_seq_list = []
        code_seq_list = []
        codes = []
        var_vocab = []
        ast_seq_list = []
        ast_seq_level_list = []
        ast_graph_list = []
        # 从根节点到所有叶子节点的路径
        ast_path_list = []

        for js in tqdm(lines):
            query = js['doc']
            query = query.replace('\n', ' ')  # 去掉回车
            query = re.sub(
                r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', query)  # 去掉括号内容
            query = query.replace('_', ' ').strip()
            query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
            query_seq_list.append(split_camel(query))  # 拆驼峰命名

            code = js['code']
            codes.append(code)
            tree = parser.parse(bytes(code, "utf8"))
            ast_seq, paths = [], []
            get_ast_seq(tree.root_node, ast_seq, paths, [])
            ast_path_list.append(paths)

            # 标记下id和parent
            #get_ast_seq_level(tree)

            var_vocab += get_vars(tree)

            ast_graph_list.append(get_ast_graph(tree, ast_vocab))

        query_seq_list = tokenizer(
            query_vocab, query_seq_list, train_query_path)

        ast_path_list = tokenizer(ast_vocab, ast_path_list, ast_path)

        save_graphs(train_ast_graph_path, ast_graph_list)

        pickle.dump(list(set(var_vocab)), open(var_vocab_path, 'wb'))
        pickle.dump(codes, open(codes_path, 'wb'))
        
    return query_seq_list, codes, var_vocab, ast_graph_list, ast_path_list
