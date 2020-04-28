import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Metrics we are using:
#
# Task 1: Infectiousness Prediction
# The first task is infectiousness prediction, where MSE is used for evaluation.
#
# Task 2: Encounter Prediction
# The second task is encounter prediction. For each user at each day, the goal is
# to predict the encounter from which the user was infected. There is a special
# encounter indicating a user was not infected.
# There are two metrics, MRR and Hit@1. MRR is mean reciprocal rank, which is computed
# as 1/|Q| \sum_{q \in Q} 1/rank_q, where |Q| is the number of total queries, and
# rank_q is the rank of the ground-truth encounter for query q.
# Hit@1 is computed as 1/|Q| \sum_{q \in Q} rank_q == 1. Basically, it measures how
# likely we rank the correct encounter at the first place.
#
# Task 3: Status Prediction
# The third task is status prediction. For each date, the goal is to identify the users
# who have been infected.
# We mainly use precision, recall and F1 for evaluation. Basically, for each date, we
# first use the ML model to predict a label for each user (exposed/infected or susceptible
# /recovered). Then we compare the list of exposed/infected users with the ground-truth
# list, and compute precision, recall and F1.
# Note that besides standard precision, we also compute precision within people who have
# no test results (precision in untested users), and precision within people who have no
# test results and no symptoms (precision in untested and asymptomatic users).
# In addition, we also add precision at different positions. By default, precisions at top
# 1%, top 5%, top 10%, top 20%, top 50% are computed.

class Metrics(nn.Module):
    def __init__(self):
        super(Metrics, self).__init__()
        self.total_infectiousness_loss = 0
        self.total_infectiousness_count = 0
        self.total_encounter_mrr = 0
        self.total_encounter_hit1 = 0
        self.total_encounter_count = 0
        self.status_prediction = dict()

    def reset(self):
        self.total_infectiousness_loss = 0
        self.total_infectiousness_count = 0
        self.total_encounter_mrr = 0
        self.total_encounter_hit1 = 0
        self.total_encounter_count = 0
        self.status_prediction = dict()

    def update(self, model_input, model_output):
        # Task 1: Infectiousness Prediction
        prediction = (
            model_output.latent_variable[:, :, 0:1]
            * model_input["valid_history_mask"][..., None]
        )
        target = (
            model_input.infectiousness_history
            * model_input["valid_history_mask"][..., None]
        )
        diff = prediction.view(
            -1
        ) - target.view(-1)
        self.total_infectiousness_loss += torch.sum(diff * diff).item()
        self.total_infectiousness_count += diff.size(0)

        # Task 2: Encounter Contagion Prediction
        # Extract prediction from model_output
        prediction = model_output.encounter_variables.squeeze(2).masked_fill(
            (1 - model_input.mask).bool(), -float("inf")
        )
        # Extract prediction from model_input
        target = model_input.encounter_is_contagion.squeeze(2)
        for k in range(target.size(0)):
            # Find position of target encounter
            label = (target[k] == 1).nonzero()
            if label.squeeze().size() != torch.Size([]):
                ranking = (prediction[k] >= 0).sum() + 1
            else:
                ranking = (prediction[k] >= prediction[k][label.item()]).sum() + 1
            self.total_encounter_mrr += 1.0 / ranking.item()
            if ranking.item() == 1:
                self.total_encounter_hit1 += 1.0
            self.total_encounter_count += 1

        # Task 3: Status Prediction
        for k in range(model_input.human_idx.size(0)):
            human_idx = model_input.human_idx[k].item()
            day_idx = model_input.day_idx[k].item()
            prediction = model_output.latent_variable[k, 0, 0].item()
            if model_input.current_compartment[k][1] == 1 or model_input.current_compartment[k][2] == 1:
                target = 1
            else:
                target = 0
            if model_input.health_history[k, :, 0:-1].sum() != 0:
                symptom = 1
            else:
                symptom = 0
            if model_input.health_history[k, :, -1].sum() != 0:
                tested = 1
            else:
                tested = 0
            if day_idx not in self.status_prediction:
                self.status_prediction[day_idx] = list()
            self.status_prediction[day_idx].append((human_idx, prediction, target, symptom, tested))

    def evaluate(self, threshold=0.1, top_n=[0.01, 0.05, 0.1, 0.2, 0.5]):
        precision, recall, f1 = 0.0, 0.0, 0.0
        precision_untested = 0.0
        precision_untested_asymptomatic = 0.0
        precision_topn = [0.0 for n in top_n]
        
        for day_idx, status in self.status_prediction.items():
            # update precision
            a, b, current_precision = 0.0, 0.0, 0.0
            for human_idx, prediction, target, symptom, tested in status:
                if prediction > threshold:
                    b += 1
                    if target == 1:
                        a += 1
            if b != 0:
                current_precision += a / b
            precision += current_precision

            # update precision for untested people
            a, b, current_precision_untested = 0.0, 0.0, 0.0
            for human_idx, prediction, target, symptom, tested in status:
                if prediction > threshold and tested == 0:
                    b += 1
                    if target == 1:
                        a += 1
            if b != 0:
                current_precision_untested += a / b
            precision_untested += current_precision_untested

            # update precision for untested and asymptomatic people
            a, b, current_precision_untested_asymptomatic = 0.0, 0.0, 0.0
            for human_idx, prediction, target, symptom, tested in status:
                if prediction > threshold and tested == 0 and symptom == 0:
                    b += 1
                    if target == 1:
                        a += 1
            if b != 0:
                current_precision_untested_asymptomatic += a / b
            precision_untested_asymptomatic += current_precision_untested_asymptomatic

            # update recall
            a, b, current_recall = 0.0, 0.0, 0.0
            for human_idx, prediction, target, symptom, tested in status:
                if target == 1:
                    b += 1
                    if prediction > threshold:
                        a += 1
            if b != 0:
                current_recall += a / b
            recall += current_recall

            # update precision@top_n
            rank_list = sorted(status, key=lambda x:x[1], reverse=True)
            for k, percentage in enumerate(top_n):
                a, b, current_precision_topn = 0.0, 0.0, 0.0
                n = int(percentage * len(rank_list))
                for i in range(n):
                    b += 1
                    if rank_list[i][2] == 1:
                        a += 1
                if b != 0:
                    precision_topn[k] += a / b

            # update f1            
            f1 += 2 * current_precision * current_recall / (current_precision + current_recall + 1e-10) 

        precision /= len(self.status_prediction)
        precision_untested /= len(self.status_prediction)
        precision_untested_asymptomatic /= len(self.status_prediction)
        recall /= len(self.status_prediction)
        f1 /= len(self.status_prediction)
        for k in range(len(precision_topn)):
            precision_topn[k] /= len(self.status_prediction)

        result = dict(
            [
                (
                    "mse",
                    math.sqrt(
                        self.total_infectiousness_loss
                        / (self.total_infectiousness_count + 0.001)
                    ),
                ),
                ("mrr", self.total_encounter_mrr / self.total_encounter_count),
                ("hit@1", self.total_encounter_hit1 / self.total_encounter_count),
                ("precision", precision),
                ("precision in untested users", precision_untested),
                ("precision in untested and asymptomatic users", precision_untested_asymptomatic),
                ("recall", recall),
                ("f1", f1)
            ]
        )
        for n, precision in zip(top_n, precision_topn):
            result['precision@top_{}'.format(n)] = precision

        return result