# Knowledge Graph Completion

## Prerequisite

* Python 3.6

* Pytorch 1.7

* Geoopt

* tqdm

## Experiments

### 1. Data Preparation

To preprocess the dataset, run the following command

```
python preprocess.py
```

### 2. Training & Testing

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
            --learning_rate=1e-4 \\
            --val_every=25 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the SPD_6^{le} model, run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tgspdle \\
            --metric=riem \\
            --dims=10 \\
            --learning_rate=1e-4 \\
            --val_every=25 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the Gr_{20,10} model, run the following command

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
            --learning_rate=1e-4 \\
            --val_every=25 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```

To train and test the SPD_5^{le} \times Gr_{5,2} model, run the following command

```
python -m torch.distributed.launch --nproc_per_node=N_CPUS --master_port=2055 train.py \\
            --n_procs=N_CPUS \\
            --data=WN18RR \\
            --run_id=test \\
            --results_file=out/results.csv \\
            --model=tgspdlegr \\
            --metric=riem \\
            --dims=5 \\
            --pdim=5 \\
            --kdim=2 \\
            --learning_rate=1e-4 \\
            --val_every=25 \\
            --patience=500 \\
            --batch_size=4096 \\
            --epochs=5000 \\
            --train_bias
```


### 3. Models and Metrics

The available options for the parameter `--model` are:

* `tgspdle`: Applies a scaling on the head embedding. Embeddings are SPD matrices with a Log-Euclidean geometry. 

* `tggr`: Applies a scaling on the head embedding. Embedding are p-dimensional subspaces of R^n. 

* `tgspdlegr`: Applies a scaling on the head embedding with the mixture model. 

In the current version, the only option for the parameter `--metric` is 

* `riem`: Riemannian metric

## Acknowledgement

We thank the authors of [Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices](https://arxiv.org/pdf/2110.13475.pdf) for releasing their code. The code in this repository is based on their source code release ([link](https://github.com/fedelopez77/gyrospd)). If you find this code useful, please consider citing their work.
