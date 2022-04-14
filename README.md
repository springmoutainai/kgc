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

### 3. Training & Testing

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=PREP \\
            --run_id=RUN_ID \\
            --results_file=out/results.csv \\
            --model=MODEL \\
            --metric=riem \\
            --dims=10 \\
            --pdim=10 \\
            --kdim=10 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the ![\operatorname{SPD}_{14}^{le}](https://latex.codecogs.com/svg.image?\operatorname{SPD}_{14}^{le}) model, 
run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tgspdle \\
            --metric=riem \\
            --dims=14 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the ![\operatorname{Gr}_{20,10}](https://latex.codecogs.com/svg.image?\operatorname{Gr}_{20,10}) model, 
run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tggr \\
            --metric=riem \\
            --dims=20 \\
            --pdim=10 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the ![\operatorname{SPD}_{13}^{le} \times \operatorname{Gr}_{14,13}](https://latex.codecogs.com/svg.image?\operatorname{SPD}_{13}^{le}&space;\times&space;\operatorname{Gr}_{14,13}) model, run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tgspdlegr \\
            --metric=riem \\
            --dims=14 \\
            --pdim=13 \\
            --kdim=13 \\
            --learning_rate=1e-3 \\
            --val_every=5 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```


### 4. Models and Metrics

The available options for the parameter `--model` are:

* `tgspdle`: Applies a scaling on the head embedding. Embeddings are SPD matrices with a Log-Euclidean geometry. 

* `tggr`: Applies a scaling on the head embedding. Embedding are p-dimensional subspaces of ![\mathbb{R}^n](https://latex.codecogs.com/svg.image?\mathbb{R}^n). 

* `tgspdlegr`: Applies a scaling on the head embedding with the mixture model. 

In the current version, the only option for the parameter `--metric` is 

* `riem`: Riemannian metric

## Acknowledgement

We thank the authors of [Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices](https://arxiv.org/pdf/2110.13475.pdf) for releasing their code. The code in this repository is based on their source code release ([link](https://github.com/fedelopez77/gyrospd)). If you find this code useful, please consider citing their work.
