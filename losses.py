from functools import reduce
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EntityMasker


def get_class(key):
    KEY_CLASS_MAPPING = {
        "infectiousness": InfectiousnessLoss,
        "contagion": ContagionLoss,
    }
    return KEY_CLASS_MAPPING[key]


class InfectiousnessLoss(nn.Module):
    def __init__(self):
        super(InfectiousnessLoss, self).__init__()
        self.masker = EntityMasker()

    def forward(self, model_input, model_output):
        assert model_output.latent_variable.dim() == 3, (
            "Infectiousness Loss can only be used on (temporal) "
            "set-valued latent variables."
        )
        # This will block gradients to the entities that are invalid
        prediction = self.masker(
            model_output.latent_variable[:, :, 0:1], model_input["valid_history_mask"]
        )
        with torch.no_grad():
            target = self.masker(
                model_input.infectiousness_history, model_input["valid_history_mask"]
            )
        return F.mse_loss(prediction, target)


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
        self.masker = EntityMasker()

    def forward(self, model_input, model_output):
        contagion_logit = model_output.encounter_variables[:, :, 0:1]
        # Mask with masker (this blocks gradients by multiplying it with 0)
        self.masker(contagion_logit, model_input.mask)
        B, M, C = contagion_logit.shape
        if self.allow_multiple_exposures:
            # encounter_variables.shape = BM1
            return F.binary_cross_entropy_with_logits(
                contagion_logit, model_input.encounter_is_contagion,
            )
        else:
            # Now, one of the encounters could have been the exposure event -- or not.
            # To account for this, we use a little trick and append a 0-logit to the
            # encounter variables before passing through a softmax. This 0-logit acts
            # as a logit sink, and enables us to avoid an extra pooling operation in
            # the transformer architecture.
            logit_sink = torch.zeros(
                (B, 1), dtype=contagion_logit.dtype, device=contagion_logit.device,
            )
            # full_logit.shape = B(1+M)
            full_logit = torch.cat([logit_sink, contagion_logit[:, :, 0]], dim=1)
            target_onehots = self._prepare_single_exposure_targets(
                model_input.encounter_is_contagion
            )
            # Now compute the softmax loss
            return F.cross_entropy(full_logit, target_onehots)

    @staticmethod
    def _prepare_single_exposure_targets(target_onehots):
        if target_onehots.dim() == 3:
            target_onehots = target_onehots[:, :, 0]
        assert target_onehots.dim() == 2
        # none_hot_mask.shape = (B,)
        nonehot_mask = torch.eq(target_onehots.max(1).values, 0.0)
        # We add the 1 because all index is moved one element to the right due to
        # the logit sink in the `full_logit`.
        target_idxs = torch.argmax(target_onehots, dim=1) + 1
        # Set the target_idx to 0 where none-hot
        target_idxs[nonehot_mask] = 0
        return target_idxs


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
