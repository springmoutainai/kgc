# Knowledge Graph Completion

## Prerequisite

* Python 3.6

* Pytorch 1.7

* [Geoopt](https://github.com/geoopt/geoopt)

* tqdm

## Experiments

### 1. Init repository

To uncompress the data and create the output directories, run the following command

```
sh init.sh
```

The dataset is included in the source code from this [link](https://github.com/fedelopez77/gyrospd), 
which is taken from this [link](https://github.com/villmow/datasets_knowledge_embedding).  

### 2. Data Preparation

To preprocess the dataset, run the following command

```
python preprocess.py
```

## Acknowledgement

We thank the authors of [Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices](https://arxiv.org/pdf/2110.13475.pdf) for releasing their code. The code in this repository is based on their source code release ([link](https://github.com/fedelopez77/gyrospd)). If you find this code useful, please consider citing their work.
