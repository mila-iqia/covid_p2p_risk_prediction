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
# We mainly use precision@k, recall@k for evaluation. Basically, for each date:
# (1) we first use ML model to predict the probability each user has been infected;
# (2) then we sort all the users based on the infection probability;
# (3) finally we compute precision and recall at top k percentage.
# In this process, we evaluate on three different sets of users:
# (1) all users;
# (2) users who have not been tested;
# (3) users who have not been tested and have not reported any symptoms.


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
        diff = prediction.view(-1) - target.view(-1)
        self.total_infectiousness_loss += torch.sum(diff * diff).item()
        self.total_infectiousness_count += diff.size(0)

        # Task 2: Encounter Contagion Prediction
        # Extract prediction from model_output
        prediction = model_output.encounter_variables.squeeze(2).masked_fill(
            (1 - model_input.mask).bool(), -float("inf")
        )
        logit_sink = torch.zeros(
            (prediction.size(0), 1), dtype=prediction.dtype, device=prediction.device,
        )
        prediction = torch.cat([prediction, logit_sink], dim=1)
        # Extract prediction from model_input
        target = model_input.encounter_is_contagion.squeeze(2)
        for k in range(target.size(0)):
            # Find position of target encounter
            label = (target[k] == 1).nonzero()
            if label.squeeze().size() != torch.Size([]):
                label = -1
            else:
                label = label.item()
            ranking = (prediction[k] >= prediction[k, label]).sum()
            self.total_encounter_mrr += 1.0 / ranking.item()
            if ranking.item() == 1:
                self.total_encounter_hit1 += 1.0
            self.total_encounter_count += 1

        # Task 3: Status Prediction
        prediction = model_output.encounter_variables.squeeze(2).masked_fill(
            (1 - model_input.mask).bool(), -float("inf")
        )
        logit_sink = torch.zeros(
            (prediction.size(0), 1), dtype=prediction.dtype, device=prediction.device,
        )
        prediction = torch.cat([prediction, logit_sink], dim=1)
        prediction = torch.softmax(prediction, dim=-1)
        for k in range(model_input.human_idx.size(0)):
            human_idx = model_input.human_idx[k].item()
            day_idx = model_input.day_idx[k].item()
            probability = 1 - prediction[k, -1].item()
            if (
                model_input.current_compartment[k][1] == 1
                or model_input.current_compartment[k][2] == 1
            ):
                infected = 1
            else:
                infected = 0
            if model_input.health_history[k, :, 0:-1].sum() != 0:
                symptomatic = 1
            else:
                symptomatic = 0
            if model_input.health_history[k, :, -1].sum() != 0:
                tested = 1
            else:
                tested = 0
            if day_idx not in self.status_prediction:
                self.status_prediction[day_idx] = list()
            self.status_prediction[day_idx].append(
                (human_idx, probability, infected, symptomatic, tested)
            )

    def compute_pr(self, rank_list, percentage):
        top_n = int(percentage * len(rank_list))
        a, b, current_precision = 0.0, 0.0, 0.0
        for i in range(top_n):
            infected, symptomatic, tested = rank_list[i][2:]
            if infected == 1:
                a += 1
            b += 1
        if b != 0:
            current_precision = a / b

        top_n = int(percentage * len(rank_list))
        a, b, current_recall = 0.0, 0.0, 0.0
        for i in range(len(rank_list)):
            infected, symptomatic, tested = rank_list[i][2:]
            if infected == 1:
                if i < top_n:
                    a += 1
                b += 1
        if b != 0:
            current_recall = a / b

        return current_precision, current_recall

    def evaluate(self, percentage_list=[0.01]):
        precision_all = [0.0 for _ in percentage_list]
        precision_nottested = [0.0 for _ in percentage_list]
        precision_nottested_notsymptomatic = [0.0 for _ in percentage_list]

        recall_all = [0.0 for _ in percentage_list]
        recall_nottested = [0.0 for _ in percentage_list]
        recall_nottested_notsymptomatic = [0.0 for _ in percentage_list]

        for day_idx, status in self.status_prediction.items():
            sorted_list = sorted(status, key=lambda x: x[1], reverse=True)

            # update precision and recall
            for k, percentage in enumerate(percentage_list):
                # precision and recall for all users
                rank_list = [item for item in sorted_list]
                precision, recall = self.compute_pr(rank_list, percentage)
                precision_all[k] += precision
                recall_all[k] += recall

                # precision and recall for not tested users
                rank_list = [item for item in sorted_list if item[3] == 0]
                precision, recall = self.compute_pr(rank_list, percentage)
                precision_nottested[k] += precision
                recall_nottested[k] += recall

                # precision and recall for not tested and not symtomatic users
                rank_list = [
                    item for item in sorted_list if (item[3] == 0 and item[4] == 0)
                ]
                precision, recall = self.compute_pr(rank_list, percentage)
                precision_nottested_notsymptomatic[k] += precision
                recall_nottested_notsymptomatic[k] += recall

        for k in range(len(precision_all)):
            precision_all[k] /= len(self.status_prediction)
        for k in range(len(precision_nottested)):
            precision_nottested[k] /= len(self.status_prediction)
        for k in range(len(precision_nottested_notsymptomatic)):
            precision_nottested_notsymptomatic[k] /= len(self.status_prediction)

        for k in range(len(recall_all)):
            recall_all[k] /= len(self.status_prediction)
        for k in range(len(recall_nottested)):
            recall_nottested[k] /= len(self.status_prediction)
        for k in range(len(recall_nottested_notsymptomatic)):
            recall_nottested_notsymptomatic[k] /= len(self.status_prediction)

        result = dict()
        result["mse"] = math.sqrt(
            self.total_infectiousness_loss / (self.total_infectiousness_count + 0.001)
        )
        result["mrr"] = self.total_encounter_mrr / self.total_encounter_count
        result["hit@1"] = self.total_encounter_hit1 / self.total_encounter_count
        for percentage, precision in zip(percentage_list, precision_all):
            result["precision top_{} all_users".format(percentage)] = precision
        for percentage, precision in zip(percentage_list, precision_nottested):
            result["precision top_{} users_not_tested".format(percentage)] = precision
        for percentage, precision in zip(
            percentage_list, precision_nottested_notsymptomatic
        ):
            result[
                "precision top_{} users_not_tested_and_no_symptoms".format(percentage)
            ] = precision
        for percentage, recall in zip(percentage_list, recall_all):
            result["recall top_{} all_users".format(percentage)] = recall
        for percentage, recall in zip(percentage_list, recall_nottested):
            result["recall top_{} users_not_tested".format(percentage)] = recall
        for percentage, recall in zip(percentage_list, recall_nottested_notsymptomatic):
            result[
                "recall top_{} users_not_tested_and_no_symptoms".format(percentage)
            ] = recall

        return result
