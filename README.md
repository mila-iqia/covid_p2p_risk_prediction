# Supervised COVID-19 risk prediction

As part of a project for creating a [COVID-19 risk-management app](https://TODO), we have created a supervised learning dataset for predicting individuals' level of risk of infection, as well as their source of infection, from features of individuals (e.g. pre-existing medical conditions) and features of encounters between individuals. The dataset is output by [a city-level simulator](https://https://github.com/pg2455/covid_p2p_simulation) (a stochastic agent-based model). 

**The goal** of providing this dataset is to find machine learning models (or any method!) which can do a good job of predicting risk and sources of infection from the provided features. The features are constrained by many concerns about privacy and security, making ordinary contact-tracing impracticable; this is why we need to train the predictors on simulated data. The simulated data is parsed to 'look like' the real data that would eventually be gathered by the app. The best risk estimator(s) will be used in an app to provide personalized recommendations and interventions. There is potential for these targetted interventions to reduce the spread of COVID-19 much more effectively than generic social distancing or other measures.

This repo contains pytorch dataloaders and a Transformer model; you can start from these and replace the Transformer with your own model, or use them as inspiration for development with another framework. Upload your results to the table by making a PR (details below). 

**IMPORTANT:** Do not train/tune on the test set, optimize for any of the metrics, or otherwise attempt to "cheat" at the task. This is not a contest. This project has real-world applications; under-estimating risk due to poor generalization/over-fitting could be dangerous.  We will keep a private test set to check for this, but is extremely important to use all machine learning best-practices, and it is everyone's individual responsibility to to so to the best of their ability.


## Quick Start

### Get the data 

Extract the provided zip file into `\data`.
```
unzip data.zip data
```

### Dependencies

Besides `pytorch` and the usual ML stack, you will need `speedrun` which you can install as: 
```
 pip install git+https://github.com/inferno-pytorch/speedrun.git@dev`. 
```

For logging with wandb, you'll also need to `pip install wandb`. 

### Train the transformer model

Make an experimental directory: `mkdir exp`

Run the training script, logging to `exp/`:
```
python train.py exp/MY-CTT-EXPERIMENT-0 --inherit base_config/CTT-0
```
This will start training on a GPU, if available. If you want to use a CPU instead, append  `--config.device cpu` to the above command. 

### Train your own model

Replace the models.py with your own if you want to use this code as a scaffold. Feel free to use only the data loaders and metrics and write your own main loop etc., but we may be slower to evaluate your PR the more different it is from this code.

## Task Details

For a full write-up of this task, see [this document](TODO).

**Input:** TODO

**Targets:** TODO

**Metrics:** 
* MSE is Mean Squared Error, between the target risk and the prediction
* MRR is  


## Results table

Model Name | Brief description | ML? | MSE | MRR
--- | --- | --- | --- | ---
[Naive Contact Tracing](TODO) | Simple risk calculation based on number of contacts | No | - | -
[Transformer](https://github.com/nasimrahaman/ctt) | Uses attention over last 14 days of encounters | Yes | - | -


## Reporting Results

To report results in the leaderboard submit a pull request from your repo to the master branch of this repo:
* Place your row at the appropriate height so that the table is sorted by performance on the first metric (MSE)
* You must fill all fields in the leaderboard row:
    - Model name (which is a link to your repo)
    - One-line description of your model
    - Whether your method employs machine learning (yes/no)
    - Metrics (MSE and MRR)
* Make sure your repo has a brief description of your model in the README.md
* The repo making the PR should contain all of your code, which must be open-source (not private)
* Tag @teganmaharaj and @nasimrahaman as reviewers of your PR


## Links to other parts of this risk-management project

The next stage for a successful risk predictor is to be integrated into the loop of the simulator to test different intervention strategies based on the predicted risk. There is also the potential to jointly train both the simulator and risk predictor. Projects linked below explore both of these possibilities.

* [Simulator](https://github.com/pg2455/covid_p2p_simulation): Generates the underlying data, using an epidemiolgically-informed agent-based model written in Simpy 
* [Data parser](TODO): Takes the logs from the Simulator and generates the supervised learning dataset, masking/discretizing some values in accordance with privacy concerns.
* [Transformer model](https://github.com/nasimrahaman/ctt): Full version of the model the example code in here is based on
* [GraphNN model](TODO): Implementation of a graph neural network for joint risk prediction and simulator optimizatio 
* [coVAEd model](TODO): Variational auto-encoder approach for joint risk prediction and model optimization
* [Decentralized Loopy Belief](TODO): Training of a graphical model of disease propagation via message-passing
