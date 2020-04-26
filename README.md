# Transformer Model for Contact Tracing

This repository contains code to train machine learning models for risk prediction. 

## Getting Started

Besides `pytorch` and the usual ML stack, you will need `speedrun` which you can install as: 
```
 pip install git+https://github.com/inferno-pytorch/speedrun.git@dev`. 
```

For logging with wandb, you'll also need to `pip install wandb`. 

Download the dataset from here, and decompress in to `data`. 

To train a model, first `mkdir exp` and then: 

```
python train.py exp/MY-CTT-EXPERIMENT-0 --inherit base_config/CTT-0
```

This will start training on a GPU, if available. If you want to use a CPU instead, append the `--config.device cpu` to the above command. 
