from functools import reduce
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_class(key):
    KEY_CLASS_MAPPING = {
        "infectiousness": InfectiousnessLoss,
        "contagion": ContagionLoss,
    }
    return KEY_CLASS_MAPPING[key]


class InfectiousnessLoss(nn.Module):
    def forward(self, model_input, model_output):
        assert model_output.latent_variable.dim() == 3, (
            "Infectiousness Loss can only be used on (temporal) "
            "set-valued latent variables."
        )
        return F.mse_loss(
            model_output.latent_variable[:, :, 0:1], model_input.infectiousness_history
        )


class ContagionLoss(nn.Module):
    def __init__(self, allow_multiple_exposures=True):
        """
        Parameters
        ----------
        allow_multiple_exposures : bool
            If this is set to False, only one encounter can be the contagion,
            in which case, we use a softmax + cross-entropy loss. If set to True,
            multiple events can be contagions, in which case we use sigmoid +
            binary cross entropy loss.
        """
        super(ContagionLoss, self).__init__()
        self.allow_multiple_exposures = allow_multiple_exposures

    def forward(self, model_input, model_output):
        if self.allow_multiple_exposures:
            return F.binary_cross_entropy_with_logits(
                model_output.encounter_variables, model_input.encounter_is_contagion
            )
        else:
            # TODO
            raise NotImplementedError


class WeightedSum(nn.Module):
    def __init__(self, losses: dict, weights: dict = None):
        super(WeightedSum, self).__init__()
        self.losses = nn.ModuleDict(losses)
        if weights is None:
            # noinspection PyUnresolvedReferences
            weights = {key: 1.0 for key in self.losses.keys()}
        self.weights = weights
        # noinspection PyTypeChecker
        assert len(self.losses) == len(self.weights)

    def forward(self, model_input, model_output):
        # noinspection PyUnresolvedReferences
        unweighted_losses = {
            key: loss(model_input, model_output) for key, loss in self.losses.items()
        }
        weighted_losses = {
            key: self.weights[key] * loss for key, loss in unweighted_losses.items()
        }
        output = Dict()
        output.unweighted_losses = unweighted_losses
        output.weighted_losses = weighted_losses
        output.loss = reduce(lambda x, y: x + y, list(weighted_losses.values()))
        return output

    @classmethod
    def from_config(cls, config):
        losses = {}
        weights = {}
        for key in config["kwargs"]:
            losses[key] = get_class(key)(**config["kwargs"][key])
            weights[key] = config["weights"].get(key, 1.0)
        return cls(losses=losses, weights=weights)
