# COVI-ml

This repository provides models, infrastructure and datasets for training deep-learning based predictors of COVID-19 infectiousness as used in [Proactive Contact Tracing](TODO). 

## What's in the Box

### Models
We provide [architectural scaffolding](ctt/models/transformers/msn.py#L36) around deep-sets (DS), set-transformers (ST) and DS-ST hybrids. 

### Training Infrastructure
The core training infrastructure is built with [speedrun](https://github.com/inferno-pytorch/speedrun). It supports experiment tracking with [Weights and Biases](https://www.wandb.com) & [tensorboard](https://www.tensorflow.org/tensorboard), and hyperparameter sweeps with [Weights and Biases Sweeps](https://www.wandb.com/sweeps).

### Datasets
The training data is derived from [COVI-sim](TODO), an agent based simulator built with contact-tracing benchmarking and epidemiological realism in mind. We include a dataset (download here), but you can print your own datasets by following instructions in the [COVI-sim repository].

## Installation

Install the dependencies with:

```
pip install -r requirements.txt
```
