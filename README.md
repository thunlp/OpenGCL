# Readme

OpenGCL is an open-source toolkit, which implements our modularized graph contrastive learning framework. Users can combine different encoders (with readout functions), discriminators, estimators and samplers by one command line, which helps them find well-performing combinations on different tasks and datasets.

## Get started

### Prerequisites

Install the following packages beforehand:

1. Python 3.7 or above;
2. Pytorch 1.7.0 or above (follow installation guide [here](https://pytorch.org/get-started/locally/));
3. Pytorch-geometric (follow installation guide [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html));
4. Other requirements in [requirements.txt](requirements.txt).

### Introduction

For the simplest use of OpenGCL, you might want to reconstruct a baseline model under our framework. Make sure your working directory is `OpenGCL/src`. Try the following command as an example:

```bash
python -m opengcl --task node --dataset cora --enc gcn --dec inner --est jsd --sampler snr --output-dim 64  --hidden-size 2 --learning-rate 0.01 --epochs 500 --early-stopping 20 --patience 3 --clf-ratio 0.2
```

This trains a GAE model on Cora, and performs node classification. 

The output should look like this:

```
[main] (2.2810919284820557s) Loading dataset...
[datasets] Loading Cora Dataset from root dir: .../OpenGCL/data/Cora
[datasets] Downloading dataloaders "Cora" from "https://github.com/kimiyoung/planetoid/raw/master/data".
[datasets] Files will be saved to "../data/Cora".
[downloader] Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.x
...
[downloader] Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index
[datasets] Dataset successfully downloaded.
[datasets] This is a medium-sized sparse dataset with 1 graph, 2708 nodes and 10556 edges.
[main] (23.69194722175598s) Splitting dataset...
[main] (23.72783327102661s) Dataset loaded and split.
[main] (23.727877616882324s) Start building GCL framework...
[GCL] Building framework... done in 0.19335460662841797s.
[GCL] Start training for 500 epochs...
[GCL]   epoch 5: loss: 1.28604; time used = 1.39682936668396s
...
[GCL]   epoch 180: loss: 0.87510; time used = 0.9593579769134521s
[GCL] Early stopping condition satisfied. Abort training.
[GCL] Framework training finished in 37.25677442550659s.
[main] (61.178184032440186s) GCL framework built and trained.
[main] (61.17819690704346s) Start classification...
[main] (66.95774221420288s) Classification finished.
[main] F1-score: micro = 0.7563451776649746.
```

After the task is finished, you should be able to find a log file in `OpenGCL/logs`, named as `YYYYmmdd_HHMMSS_X_cora_node.txt`, which writes something similar to

```
python3 -m opengcl --task node --dataset cora --enc gcn --dec inner --est jsd --sampler snr --output-dim 64 --hidden-size 2 --learning-rate 0.01 --epochs 500 --early-stopping 20 --patience 3 --clf-ratio 0.2
0.7563451776649746
```

## Usage

In `OpenGCL/src`, run OpenGCL by the following command:

```bash
python -m opengcl --task TASK --dataset DATASET {MODULE_PARAMS} {HYPER_PARAMS}
```

with

```bash
{MODULE_PARAMS} := --enc ENCODER [--readout READOUT] --dec DISCRIMINATOR --est ESTIMATOR --sampler [SAMPLER]
{HYPER_PARAMS} := [--output-dim INTEGER] [--hidden-size INTEGER] [--dropout FLOAT] [--batch-size INTEGER] [--learning-rate FLOAT] [--epochs INTEGER] [--early-stopping INTEGER] [--patience INTEGER] [--clf-ratio FLOAT]
```

We will discuss each parameter in detail in the following subsections.

### Task

Choose one downstream task for GCL. Choose `node` for node classification, or `graph` for graph classification.

#### Range

- `--task {node, graph}`: two classification tasks

### Dataset

Choose one dataset for GCL. We provide 13 datasets, 5 of which are multi-graph, and 8 are single-graph datasets.

#### Range

- `--task {cora, citeseer, pubmed, amazon_computers, amazon_photo, coauthor_cs, coauthor_phy, wikics}`: single-graph datasets
- `--task {reddit_binary, imdb_binary, imdb_multi, mutag, ptc_mr}`: multi-graph datasets.

### Module parameters

Select encoder, decoder, estimator and sampler that make up the model.

#### Encoder

Choose GNN encoders (`gcn`, `gin` and `gat`), MLP encoder (`linear`), or lookup table (`none`) for encoder.

Encoders are single or multiple consecutive layers that translate node features to node embedding vectors. Lookup table, of course, has only one layer. 

##### Range

- `--enc {gcn, gin, gat, linear, none}`

##### Related parameters

- `--output-dim`: output dimension of each layer.
- `--hidden-size`: number of hidden layers. Total layer number should be `HIDDEN_SIZE + 1`. The dimensions of the input layer is `(feature_dim, output_dim)`, while all the following layers have dimensions `(output_dim, output_dim)`.

#### Readout

To get graph embedding, node vectors should bypass a readout function. Choose from `sum` pooling, `mean` pooling or `jk-net`.

##### Range

- `[--readout {sum, mean, jk-net}]`
- Optional; not needed (ignored) on some occasions. The default value is `mean`.

##### Related parameters

- `--task`: for graph classification, a readout function is needed to get graph embeddings.
- `--sampler`: for samplers that require global view, graph embeddings are needed.

#### Discriminator

Choose one discriminator from inner product (`inner`), bilinear product (`bilinear`).

Discriminators calculate likelihood for sample pairs.

##### Range

- `--dec {inner, bilinear}`

#### Estimator

Choose one estimator from Jensen-Shannon Divergence (`jsd`) or InfoNCE (`nce`) loss.

Estimators calculate loss for likelihood scores of (anchor, positive) and (anchor, negative) pairs.

##### Range

- `--est {jsd, nce}`

#### Sampler

Choose from single-view samplers (`s_` samplers) and multi-view samplers (`dgi`, `mvgrl`, `gca`, `graphcl`).

Samplers are described as below. Views of (anchors - context samples) are listed in 'View(s)'.  

| Sampler   | Source         | Description                                                  | View(s)                            |
| --------- | -------------- | ------------------------------------------------------------ | ---------------------------------- |
| `snr`     | LINE, GAE      | anc, pos: neighbors; neg: random sample                      | single; local-local                |
| `srr`     | DeepWalk       | anc, pos: random walk; neg: random sample                    | single; local-local                |
| `dgi`     | DGI, InfoGraph | anc: graph G; pos: nodes in G; neg: nodes in other graphs    | multi; global-local                |
| `mvgrl`   | MVGRL          | anc: graph; pos: nodes in G diffused; neg: nodes in other diffused graphs | multi; global-local; augmentative  |
| `gca`     | GCA            | two augmentations G1, G2; anc, pos: corresponding nodes in G1, G2; neg: all other nodes | multi; local-local; augmentative   |
| `graphcl` | GraphCL        | two augmentations G1, G2; anc, pos: G1, G2; neg: another graph | multi; global-global; augmentative |

##### Range

- `--sampler {snr, srr, dgi, mvgrl, gca, graphcl}`

### Hyperparameters

Hyperparameters are structural or training parameters of the model or training process. These are optional parameters; default values are given. We list the hyperparameters, their meanings, type, range and default values in the following table.

| Parameter          | Meaning                                                      | Type       | Range            | Default |
| ------------------ | ------------------------------------------------------------ | ---------- | ---------------- | ------- |
| `--output-dim`     | output dimension of embeddings                               | structural | positive integer | 64      |
| `--hidden-size`    | number of hidden layers                                      | structural | natural number   | 0       |
| `--dropout`        | dropout factor while training                                | training   | float in [0, 1)  | 0.      |
| `--batch-size`     | number of samples in each batch                              | training   | positive integer | 4096    |
| `--learning-rate`  | learning rate                                                | training   | positive float   | 0.01    |
| `--epochs`         | number of training epochs                                    | training   | natural number   | 500     |
| `--early-stopping` | minimum epoch for early stopping                             | training   | natural number   | 20      |
| `--patience`       | early stop after `PATIENCE` epochs of consecutive loss growth | training   | natural number   | 3       |
| `--clf-ratio`      | ratio of dataset used to train classifier                    | training   | float in (0, 1)  | 0.5     |

## Baselines

We list example commands of baseline models here. We use Cora for node classification and MUTAG for graph classification.

| Name      | Task  | Command                                                      |
| --------- | ----- | ------------------------------------------------------------ |
| GAE       | node  | `python -m opengcl --task node --dataset cora --enc gcn --dec inner --est jsd --sampler snr --epochs 500 --clf-ratio 0.2` |
| GCA       | node  | `python -m opengcl --task node --dataset cora --enc gcn --dec inner --est nce --sampler gca --epochs 500 --clf-ratio 0.2` |
| DGI       | node  | `python -m opengcl --task node --dataset cora --enc gcn --readout mean  --dec bilinear --est jsd --sampler dgi --epochs 500 --clf-ratio 0.2 ` |
| DGI       | graph | `python -m opengcl --task graph --dataset mutag --enc gcn --readout mean --dec bilinear --est jsd --sampler dgi --epochs 500 --clf-ratio 0.8` |
| MVGRL     | node  | `python -m opengcl --task node --dataset cora --enc gcn --readout sum --dec inner --est jsd --sampler mvgrl --epochs 500 --clf-ratio 0.2` |
| MVGRL     | graph | `python -m opengcl --task graph --dataset mutag --enc gcn --readout sum --dec inner --est jsd --sampler mvgrl --epochs 500 --clf-ratio 0.8` |
| InfoGraph | graph | `python -m opengcl --task graph --dataset mutag --enc gin --readout sum --dec inner --est jsd --sampler dgi --epochs 500 --clf-ratio 0.8` |
| GraphCL   | graph | `python -m opengcl --task graph --dataset mutag --enc gin --readout sum --dec inner --est nce --sampler graphcl --epochs 500 --clf-ratio 0.8` |

