import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Metrics(nn.Module):
    def __init__(self):
        super(Metrics, self).__init__()

        self.total_infectiousness_loss = 0
        self.total_infectiousness_count = 0
        self.total_encounter_mrr = 0
        self.total_encounter_count = 0

    def reset(self):
        self.total_infectiousness_loss = 0
        self.total_infectiousness_count = 0
        self.total_encounter_mrr = 0
        self.total_encounter_count = 0

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
                ranking = (prediction[k] > 0).sum() + 1
            else:
                ranking = (prediction[k] > prediction[k][label.item()]).sum() + 1
            self.total_encounter_mrr += 1.0 / ranking.item()
            self.total_encounter_count += 1

    def evaluate(self):
        return dict(
            [
                (
                    "mse",
                    math.sqrt(
                        self.total_infectiousness_loss
                        / (self.total_infectiousness_count + 0.001)
                    ),
                ),
                ("mrr", self.total_encounter_mrr / self.total_encounter_count),
            ]
        )
