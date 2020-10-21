# COVI-ml

This repository provides models, infrastructure and datasets for training deep-learning based predictors of COVID-19 infectiousness as used in [Proactive Contact Tracing](TODO). 

## What's in the Box

### Models
We provide [architectural scaffolding](ctt/models/transformers/msn.py#L36) around deep-sets (DS), set-transformers (ST) and DS-ST hybrids. 

### Training Infrastructure
The core training infrastructure is built with [speedrun](https://github.com/inferno-pytorch/speedrun). It supports experiment tracking with [Weights and Biases](https://www.wandb.com) & [tensorboard](https://www.tensorflow.org/tensorboard), and hyperparameter sweeps with [Weights and Biases Sweeps](https://www.wandb.com/sweeps).

### Datasets
The training data is derived from [COVI-sim](TODO), an agent based simulator built with contact-tracing benchmarking and epidemiological realism in mind. We will include a dataset (download to appear soon); but in the mean time, you can print your own datasets by following instructions in the [COVI-sim repository](TODO).

The training and validation data is structured in directories containing [zarr](https://zarr.readthedocs.io/en/stable/) datasets. There is some flexibility here (see below), but we expect two directories `./data/train` and `./data/val`, where each directory should contain an arbitrary number of zarr files.      

## Installation

This repository works with [PyTorch](https://pytorch.org/). Install the dependencies with:

```
pip install -r requirements.txt
```

Please make sure you have a [Weights and Biases account](https://wandb.ai/) and it is [configured correctly](https://docs.wandb.com/quickstart). If you do not wish to use Weights and Biases, you will need to run the following command before running the main training script: 

```
export WANDB_MODE=dryrun
```

## Training

To run your training script, first make a directory where your experiment logs will live. Once you're in this repository, 

```
mkdir experiments
``` 

If your data lives in `./data`, you may use the following command to train DS-PCT:
```
python train.py experiments/DS-PCT-0 --inherit base_configs/DS-PCT-X
```

... or the following to run ST-PCT:
```
python train.py experiments/ST-PCT-0 --inherit base_configs/ST-PCT-X
```

If your data lives elsewhere, you will need to run the following command:
```
python train.py experiments/DS-PCT-0 --inherit base_configs/DS-PCT-X --config.data.paths.train path/to/training/data --config.data.paths.validate path/to/validation/data
```

## Visualizing results

If you are using Weights and Biases, you should have a project named `ctt` under your account. Additionally, tensorboard logs are dumped in `experiments/DS-PCT-0/Logs` and checkpoints are stored in `experiments/DS-PCT-0/Weights` (likewise for ST-PCT). 

## Reporting bugs and getting help

If you find a bug or have a question, please open an issue in this repository.
 