from functools import reduce
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from ctt.models.modules import EntityMasker
from ctt.utils import typed_sum_pool


def get_class(key):
    KEY_CLASS_MAPPING = {
        "infectiousness": InfectiousnessLoss,
        "viral_load": ViralLoadLoss,
        "exposure": ExposureHistoryLoss,
        "vl2i": VL2IMultiplierLoss,
        "contagion": ContagionLoss,
    }
    return KEY_CLASS_MAPPING[key]


class SmoothedBinLoss(nn.Module):
    def __init__(self, spillage=1, gamma=2.0, reduction="mean"):
        super(SmoothedBinLoss, self).__init__()
        self.spillage = spillage
        self.gamma = gamma
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self, input, target):
        # input.shape = BMC,
        # target.shape = BM
        B, M = target.shape
        _, _, C = input.shape
        assert list(input.shape) == [B, M, C]
        with torch.no_grad():
            # noinspection PyTypeChecker
            _II, _JJ = torch.meshgrid(
                torch.arange(B, dtype=torch.long, device=target.device),
                torch.arange(M, dtype=torch.long, device=target.device),
            )
            # target.shape = BMC
            _target = torch.zeros((B, M, C), dtype=torch.float32, device=target.device)
            # Smooth it out
            _target[_II, _JJ, target] += 1.0
            for delta in range(1, self.spillage + 1):
                # fmt: off
                _target[_II, _JJ, (target - delta).clamp_min(0)] += self.gamma ** (-delta)
                _target[_II, _JJ, (target + delta).clamp_max(C)] += self.gamma ** (-delta)
                # fmt: on
            target = _target.div_(_target.sum(-1, keepdim=True))
        input = torch.log_softmax(input, dim=-1)
        loss = self.kl(input, target)
        return loss


class QuantileLoss(nn.Module):
    def __init__(self, quantiles, reduction="mean"):
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction

    def forward(self, input, target):
        # input.shape = BMC
        # target.shape = BM
        assert input.dim() == 3
        assert input.shape[-1] == len(self.quantiles)
        if target.dim() == 3:
            assert target.shape[-1] == 1
            target = target[:, :, 0]
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - input[:, :, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        # unreduced_loss.shape = BM
        unreduced_loss = torch.sum(torch.cat(losses, dim=-1), dim=-1)
        if self.reduction == "mean":
            return unreduced_loss.mean()
        elif self.reduction == "sum":
            return unreduced_loss.sum()
        elif self.reduction == "none":
            return unreduced_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}.")


class GaussianLogLikLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(GaussianLogLikLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        assert input.dim() == 3
        assert input.shape[-1] == 2
        if target.dim() == 2:
            target = target[:, :, None]
        else:
            assert target.dim() == 3
            assert target.shape[-1] == 1
        normal = td.Normal(loc=input[:, :, 0:1], scale=torch.exp(input[:, :, 1:2]))
        unreduced_loss = -(normal.log_prob(target))
        if self.reduction == "mean":
            return unreduced_loss.mean()
        elif self.reduction == "sum":
            return unreduced_loss.sum()
        elif self.reduction == "none":
            return unreduced_loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}.")


class EntityMaskedLoss(nn.Module):
    EPS = 1e-7

    def __init__(self, loss_cls):
        super(EntityMaskedLoss, self).__init__()
        self.loss_fn = loss_cls(reduction="none")
        assert isinstance(
            self.loss_fn,
            (
                nn.MSELoss,
                GaussianLogLikLoss,
                nn.BCEWithLogitsLoss,
                nn.CrossEntropyLoss,
                SmoothedBinLoss,
                QuantileLoss,
            ),
        )

    def reduce_samples(self, unreduced, sample_weights):
        if sample_weights is None:
            return unreduced.mean()
        else:
            # Broadcast sample_weights to work with all unreduced shapes, as long as
            # the first axis works
            assert sample_weights.shape[0] == unreduced.shape[0], (
                f"Shape of sample weights {sample_weights.shape} "
                f"incompatible with that of unreduced loss: {unreduced.shape}."
            )
            broadcast_shape = [sample_weights.shape[0]] + (unreduced.dim() - 1) * [1]
            sample_weights = sample_weights.reshape(broadcast_shape)
            return (unreduced * sample_weights).mean()

    def forward(self, input, target, mask, sample_weight=None):
        assert input.dim() == 3, "Input should be a BMC tensor."
        assert mask.dim() == 2, "Mask should be a BM tensor."
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            # target.shape = BM1 of integers, specifying the index of the bin.
            # Squeeze to a BM tensor.
            if target.dim() == 3:
                assert target.shape[-1] == 1
                target = target[:, :, 0]
            # input.shape = BMC of logits, but pytorch expects BCM.
            input = input.transpose(1, 2)
            # loss_elements should be a BM tensor
            loss_elements = self.loss_fn(input, target)
            # Mask out the invalids
            masked_loss_elements = loss_elements * mask
            reduced_loss = self.reduce_samples(
                masked_loss_elements.sum(-1) / (mask.sum(-1) + self.EPS), sample_weight
            )
        elif isinstance(self.loss_fn, SmoothedBinLoss):
            # target.shape = BM1 of integers specifying the index of the bin.
            # Squeeze to a BM tensor for downstream.
            if target.dim() == 3:
                assert target.shape[-1] == 1
                target = target[:, :, 0]
            # input.shape = BMC, which is what the downstream func expects.
            # loss_elements.shape = BMC, but the last dimension should be summed over
            # to get the real KL div. This gives us:
            # loss_elements.shape = BM
            loss_elements = self.loss_fn(input, target).sum(-1)
            masked_loss_elements = loss_elements * mask
            reduced_loss = self.reduce_samples(
                masked_loss_elements.sum(-1) / (mask.sum(-1) + self.EPS), sample_weight
            )
        else:
            loss_elements = self.loss_fn(input, target)
            masked_loss_elements = (
                loss_elements[..., 0] if loss_elements.dim() == 3 else loss_elements
            ) * (mask[..., 0] if mask.dim() == 3 else mask)
            reduced_loss = self.reduce_samples(
                masked_loss_elements.sum(-1) / (mask.sum(-1) + self.EPS), sample_weight,
            )
        return reduced_loss


class InfectiousnessLoss(nn.Module):
    def __init__(
        self,
        nll_loss_fn="MSELoss",
        binned=False,
        spillage=None,
        gamma=2.0,
        quantiles=None,
    ):
        super(InfectiousnessLoss, self).__init__()
        if binned:
            if spillage is None:
                self.masked_loss = EntityMaskedLoss(nn.CrossEntropyLoss)
            else:
                self.masked_loss = EntityMaskedLoss(
                    lambda reduction: SmoothedBinLoss(
                        spillage=spillage, gamma=gamma, reduction=reduction
                    )
                )
        else:
            if quantiles is None:
                loss_fn = getattr(nn, nll_loss_fn, None) or globals()[nll_loss_fn]
            else:
                loss_fn = lambda reduction: QuantileLoss(
                    quantiles=list(quantiles), reduction=reduction
                )
            self.masked_loss = EntityMaskedLoss(loss_fn)

    def forward(self, model_input, model_output):
        key = (
            "latent_variable"
            if "latent_variable" in model_output
            else "infectiousness_history"
        )
        predicted_infectiousness_history = model_output[key]
        assert predicted_infectiousness_history.dim() == 3, (
            "Infectiousness Loss can only be used on (temporal) "
            "set-valued latent variables."
        )
        # This will block gradients to the entities that are invalid
        return self.masked_loss(
            predicted_infectiousness_history,
            model_input.infectiousness_history,
            model_input["valid_history_mask"],
            model_input.get("sample_weight", None),
        )


class ViralLoadLoss(InfectiousnessLoss):
    def forward(self, model_input, model_output):
        key = (
            "latent_variable"
            if "latent_variable" in model_output
            else "viral_load_history"
        )
        predicted_viral_load_history = model_output[key]
        assert predicted_viral_load_history.dim() == 3, (
            "Infectiousness Loss can only be used on (temporal) "
            "set-valued latent variables."
        )
        # This will block gradients to the entities that are invalid
        return self.masked_loss(
            predicted_viral_load_history,
            model_input.viral_load_history,
            model_input["valid_history_mask"],
            model_input.get("sample_weight", None),
        )


class VL2IMultiplierLoss(nn.Module):
    def forward(self, model_input, model_output):
        predicted_vl2i_multiplier = model_output["vl2i_multiplier"][:, 0]
        target_vl2i_multiplier = model_input["vl2i_multiplier"]
        return F.mse_loss(predicted_vl2i_multiplier, target_vl2i_multiplier)


class ExposureHistoryLoss(nn.Module):
    def __init__(self):
        super(ExposureHistoryLoss, self).__init__()
        self.masked_loss = EntityMaskedLoss(nn.BCEWithLogitsLoss)

    def forward(self, model_input, model_output):
        key = (
            "latent_variable"
            if "latent_variable" in model_output
            else "exposure_history"
        )
        predicted_exposure_history = model_output[key]
        # shape = BT1
        assert predicted_exposure_history.dim() == 3, (
            "Infectiousness Loss can only be used on (temporal) "
            "set-valued latent variables."
        )
        # We could have predicted the presoftmax logits, but then the masking gets
        # tricky. Predicting presigmoid logits instead is intuitive, straightforward,
        # and faster + it should have the same effect.
        return self.masked_loss(
            predicted_exposure_history,
            model_input.exposure_history,
            model_input["valid_history_mask"],
            model_input.get("sample_weight", None),
        )


class ContagionLoss(nn.Module):
    def __init__(self, allow_multiple_exposures=True, diurnal_exposures=False):
        """
        Parameters
        ----------
        allow_multiple_exposures : bool
            If this is set to False, only one encounter can be the contagion,
            in which case, we use a softmax + cross-entropy loss. If set to True,
            multiple events can be contagions, in which case we use sigmoid +
            binary cross entropy loss.
        diurnal_exposures : bool
            If this is set to True (default: False), then we are interested in
            predicting in which of the past 14 (or however many) days there was a
            contagion encounter.
        """
        super(ContagionLoss, self).__init__()
        self.allow_multiple_exposures = allow_multiple_exposures
        self.diurnal_exposures = diurnal_exposures
        self.masked_bce = EntityMaskedLoss(nn.BCEWithLogitsLoss)
        self.masker = EntityMasker(mode="logsum")

    def forward(self, model_input, model_output):
        contagion_logit = model_output.encounter_variables[:, :, 0:1]
        if self.allow_multiple_exposures:
            if self.diurnal_exposures:
                # Convert the labels from being per-encounter to being per-day.
                contagion_labels = typed_sum_pool(
                    model_input.encounter_is_contagion,
                    type_=model_input.encounter_day,
                    reference_types=model_input.history_days,
                )
                mask = model_input.valid_history_mask
            else:
                contagion_labels = model_input.encounter_is_contagion
                mask = model_input.mask
            # encounter_variables.shape = BM1
            return self.masked_bce(contagion_logit, contagion_labels, mask)
        else:
            assert not self.diurnal_exposures
            # Mask with masker (this blocks gradients by multiplying it with 0)
            self.masker(contagion_logit, model_input.mask)
            B, M, C = contagion_logit.shape
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
