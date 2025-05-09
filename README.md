# Deep Code Search with Naming-Agnostic Contrastive Multi-view Learning

Source code for our TKDD paper "Deep Code Search with Naming-Agnostic Contrastive Multi-view Learning" [[arXiv]](https://arxiv.org/abs/2408.09345).

###  Environment

* Python 3.8
* pytorch 1.12

* transformers 2.5.0
* tree-sitter 0.20.0
* dgl 0.8.2

### Data

We use four datasets: CodeSearchNet-Python, CodeSearchNet-Java, CoSQA and CoSQA-Var.

### Command Line Parameters

`ast_pretrain/train.py` is the main entry of the AST pretrain, it requires several parameters as follows:

* num-workers: num of workers to use.
* num-copies: num of dataset copies that fit in memory.

* num-samples: num of samples per batch per worker.
* epochs: number of training epochs.
* optimizer: optimizer, (Possible values: 'sgd', 'adam', 'adagrad').
* lr_decay_epochs: where to decay lr, can be a list.
* lr_decay_rate: decay rate for learning rate.
* model: the graph neural network model used,  (Possible values: "gat", "mpnn", "gin").
* query_emb_size:  embedding size of query token.
* query_lstm_size: size of lstm embedding.
* query_hidden_size: size of final query embedding.
* ast_path_emb_size: embedding size of ast token.
* ast_path_lstm_size: size of lstm embedding.
* ast_path_hidden_size: size of final ast token embedding.
* nce-k: temperature coefficient of loss function.
* nce-t: temperature coefficient of loss function.
* positional-embedding-size: graph Laplacian vector size.
* degree-embedding-size: embedding size of degree.
* data-path: the location of the pretrained data.

`codesearch/run_with_gcc.py` is the main entry of the Code Search phase, it requires several parameters as follows:

* gcc_ratio: the proportion of graph pre-training in code search (Possible values: 0.001,0.0001...).
* ast_encode_path: the location of the pretrained model.
* train_data_file: the train dataset used in the experiment.
* valid_data_file: the valid dataset used in the experiment.
* eval_data_file: the eval dataset used in the experiment.
* retrieval_code_base: the codebase of code search used in the experiment.
* per_gpu_train_batch_size: size of batch.
* epochs: number of training epochs.

### Example

Use the following commands for pre-training and the code search task: 

```shell
python ast_pretrain/train.py
python codesearch/run_with_gcc.py
```

