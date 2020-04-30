# Supervised COVID-19 risk prediction

As part of a project for creating a COVID-19 risk-management app, we have created a supervised learning dataset for predicting individuals' level of risk of infection, as well as their source of infection, from features of individuals (e.g. pre-existing medical conditions) and features of encounters between individuals. The dataset is output by [a city-level simulator](https://github.com/pg2455/covid_p2p_simulation) (a stochastic agent-based model). 

**The goal** of providing this dataset is to find machine learning models (or any method!) which can do a good job of predicting risk and sources of infection from the provided features. The features are constrained by many concerns about privacy and security, making ordinary contact-tracing impracticable; this is why we need to train the predictors on simulated data. The simulated data is parsed to 'look like' the real data that would eventually be gathered by the app. The best risk estimator(s) will be used in an app to provide personalized recommendations and interventions. There is potential for these targetted interventions to reduce the spread of COVID-19 much more effectively than generic social distancing or other measures.

This repo contains pytorch dataloaders and a Transformer model; you can start from these and replace the Transformer with your own model, or use them as inspiration for development with another framework. Upload your results to the table by making a PR (details below). 

**IMPORTANT:** Do not train/tune on the test set, optimize for any of the metrics, or otherwise attempt to "cheat" at the task. This is not a contest. This project has real-world applications; under-estimating risk due to poor generalization/over-fitting could be dangerous.  We will keep a private test set to check for this, but is extremely important to use all machine learning best-practices, and it is everyone's individual responsibility to do so to the best of their ability.


## Quick Start / Overview

1. Clone or fork this repo
2. [Download the data](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/covi-1k-04-27.zip)
3. Extract the data to a folder called data inside the repo : `unzip covi-1k-04-27.zip data`
4. Install dependencies (see below) and `mkdir exp`
5. Run the transformer on CPU to make sure everything is working `python train.py exp/MY-CTT-EXPERIMENT-0 --inherit base_config/CTT-0 --config.device cpu` 
6. Replace the transformer model with your own and start experimenting! Also welcome to change/improve the transformer model
7. Upload your results to the results table below by making a PR to this repo

## More information

### Dataset details

Dataset ID |  Clustering Type | Target Risk Predictor | Simulator Version | Risk Prediction Version | Mobility Level | App Adoption | Population | Duration (days) | Seeds | Train/Val | Public Test 
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
V1-1K | Heuristic  |Naive First-Order Contact Tracing | [e302ecfa1fe](https://github.com/pg2455/covid_p2p_simulation/commit/e302ecfa1fe305ae766f61a8b8a7bfaf92c0d6b5)  | [bbb4124b](https://github.com/mila-iqia/covid_p2p_risk_prediction/commit/bbb4124bee1211fda23c0bfb0c1e1168db3fd73c) | Low | 50% | 1,000 | 60 | 10 | [Train/Val](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/1k_app_0.5.zip) | Coming Soon
V1-1K | Heuristic  |Naive First-Order Contact Tracing | [e302ecfa1fe](https://github.com/pg2455/covid_p2p_simulation/commit/e302ecfa1fe305ae766f61a8b8a7bfaf92c0d6b5)  | [bbb4124b](https://github.com/mila-iqia/covid_p2p_risk_prediction/commit/bbb4124bee1211fda23c0bfb0c1e1168db3fd73c) | Low | 100% | 1,000 | 60 | 10 | [Train/Val](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/1k_app_1.zip) | Coming Soon
V1-1K | Heuristic  |Naive First-Order Contact Tracing | [e302ecfa1fe](https://github.com/pg2455/covid_p2p_simulation/commit/e302ecfa1fe305ae766f61a8b8a7bfaf92c0d6b5)  | [bbb4124b](https://github.com/mila-iqia/covid_p2p_risk_prediction/commit/bbb4124bee1211fda23c0bfb0c1e1168db3fd73c) | High | 50% | 1,000 | 60 | 10 | [Train/Val](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/1k_app_0.5_feelgood.zip) | [Public Test](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/1k_app_0.5_feelgood-public-test.zip)
V1-1K | Heuristic  |Naive First-Order Contact Tracing | [e302ecfa1fe](https://github.com/pg2455/covid_p2p_simulation/commit/e302ecfa1fe305ae766f61a8b8a7bfaf92c0d6b5)  | [bbb4124b](https://github.com/mila-iqia/covid_p2p_risk_prediction/commit/bbb4124bee1211fda23c0bfb0c1e1168db3fd73c) | High | 100% | 1,000 | 60 | 10 | [Train/Val](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/1k_app_1_feelgood.zip) | [Public Test](https://covid-p2p-simulation.s3.ca-central-1.amazonaws.com/1k_app_1_feelgood-public-test.zip)
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-1K | Heuristic |Transformer | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-1K | Heuristic |Transformer | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-1K | Heuristic |Transformer | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-1K | Heuristic |Transformer | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD
V1-50K | Heuristic |Naive First-Order Contact Tracing | TBD  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD

Extract the provided zip file into `\data`.
```
unzip <data-file-name>.zip data
```

### Dependencies

To install the dependencies, simply `pip install -r requirements.txt` and you're all set. 

### Train the transformer model

Make an experimental directory: `mkdir exp`

Run the training script, logging to `exp/`:
```
python train.py exp/MY-CTT-EXPERIMENT-0 --inherit base_config/CTT-0
```
where `MY-CTT-EXPERIMENT-0` is the _experiment directory_ and you may call it anything that makes sense to you. 

The above command will start training a transformer model on a GPU if available, and dump the tensorboard logs in `exp/MY-CTT-EXPERIMENT-0/Logs`. If you have set up [Weights and Biases](https://www.wandb.com/), simply append `--config.wandb.use True` to the command. If you do not want to use GPU even if you have one available, append `--config.device cpu`. 

To get a sense of the hyperparameters you can tune, take a look inside `base_configs/CTT-0/Configurations/train_config.yml`. 

### Train your own model

Replace the models.py with your own if you want to use this code as a scaffold. Feel free to use only the data loaders and metrics and write your own main loop etc., but we may be slower to evaluate your PR the more different it is from this code.

### Task Details

This is framed as a supervised learning task, with the following inputs, targets, and metrics:

Inputs and targets are described for 1 data example (1 person), from the point of view of Alice, who has encounters with many Bobs.

**Input:** 
* `reported_symptoms`: (*14, N_s*) array where *N_s* is the number of possible symptoms. Each (*i,j*) element of the array is a binary indicator of whether Alice had symptom *j* at day *t-i*
* `test_results` (14) array of values {-1,0,1} where the *i*th element indicates the test result at day *t-i*. A value of -1 means tested negative, 0 means not tested, and 1 means tested positive.
* `candidate_encounters`: (*N_e, 3*) array where *N_e* is the number of encounters in the past 14 days. For each encounter, the 3 dimensions are: 
    - `ID`: an integer indicating the identity of the Bob in the encounter (estimated by a [clustering algorithm](https://www.overleaf.com/read/skcfppcjmccm))
    - `encounter_risk`: [0-15] the discretized risk level of the Bob in the encounter, 0 is the lowest level of risk and 15 is the highest. These risks are estimated by a model; current datasets use a naive contact-tracing calculation based on the number of Bobs encountered.
    - `day` [0-13] day of the encounter, 0 being today and 13 being 14 days ago

**Targets:** 
* Classification of infection status:
    - Binary variable 0/1 of whether Alice is infected
    - (*N_e*) binary array 0/1 for each encounter, where the target is 1 if Alice was infected by that Bob encounter and 0 if they are were not
* Regression of personal infectiousness: (14) array of floats of Alice's infectiousness for each of the past 14 days

**Metrics:** 

* **P**: Precision is of the top 1% of highest-risk people, what % are correctly identified as being infected
* **P-U**: Precision-Untested is of the top 1% of highest-risk people, excluding those who have a positive test, what % are correctly identified as being infected
* **P-A**: Precision-Asymptomatic is of the top 1% of highest-risk people, excluding those who have a positive test and those who have symptoms, what % are correctly identified as being infected
* **R**: Recall is what % of those infected are correctly identified as being infected
* **R-U**: Recall-Untested is what % of those infected are correctly identified as being infected, among people who have not been tested
* **R-A** Recall-Asymptomatic is what % of those infected are correctly identified as being infected, among people who are asymptomatic
* **MSE**: is Mean Squared Error, between the target risk and the prediction for the infectiousness of each person. (Possibly N/A for non-ML methods)
* **MRR**: is Mean Reciprocal Rank TODO


## Results 

Model Name | Brief description | ML? | P | P-U | P-A | R | R-U | R-A | MSE | MRR
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
[Naive Contact Tracing](TODO) | Simple risk calculation based on number of contacts | No | - | -| - | - | - | - | - | -
[Transformer](https://github.com/nasimrahaman/ctt) | Uses attention over last 14 days of encounters | Yes | - | - | - | - | - | - | - | -


### Reporting Results

To report results in the leaderboard submit a pull request from your repo to the master branch of this repo:
* Place your row at the appropriate height so that the table is sorted by performance on the first metric (Precision)
* You must fill all fields in the leaderboard row (except metrics which do not apply to your method:
    - Model name (which is a link to your repo)
    - One-line description of your model
    - Whether your method employs machine learning (yes/no)
    - Metrics (all Precision and Recalls)
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
