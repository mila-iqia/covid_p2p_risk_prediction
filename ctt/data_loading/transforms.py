from addict import Dict
from contextlib import contextmanager

import numpy as np
import torch

import ctt.utils as cu
from ctt.utils import Compose

# ------------------------------
# ------- Infrastructure -------
# ------------------------------


class Transform(object):
    INVERT = False

    @classmethod
    @contextmanager
    def invert_all_transforms(cls):
        old_invert = cls.INVERT
        cls.INVERT = True
        yield
        cls.INVERT = old_invert

    def apply(self, input_dict: Dict) -> Dict:
        return input_dict

    def inverse_apply(self, output_dict):
        return output_dict

    def __call__(self, io_dict: Dict) -> Dict:
        io_dict = Dict(io_dict)
        if not self.INVERT:
            return self.apply(io_dict)
        else:
            return self.inverse_apply(io_dict)


class PreTransform(object):
    def apply(
        self, human_day_info: dict, human_idx: int = None, day_idx: int = None
    ) -> dict:
        raise NotImplementedError

    def __call__(
        self, human_day_info: dict, human_idx: int = None, day_idx: int = None
    ):
        human_day_info = dict(human_day_info)
        return self.apply(human_day_info, human_idx, day_idx)


class ComposePreTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(
        self, human_day_info: dict, human_idx: int = None, day_idx: int = None
    ):
        for transform in self.transforms:
            human_day_info = transform(human_day_info, human_idx, day_idx)
        return human_day_info


# --------------------------
# ------- Transforms -------
# --------------------------


class QuantizedGaussianMessageNoise(Transform):
    def __init__(self, num_risk_levels=16, noise_std=1):
        self.num_risk_levels = num_risk_levels
        self.noise_std = noise_std

    def apply(self, input_dict: Dict) -> Dict:
        encounter_message = input_dict["encounter_message"]
        if encounter_message.shape[0] == 0:
            # No encounter messages, so nothing to do.
            return input_dict
        assert (
            encounter_message.shape[-1] == 1
        ), "Noising is only supported for float encoded messages."
        if self.noise_std == 0:
            # Shortcut path where we don't add any noise
            return input_dict
        elif self.noise_std == -1:
            # Shortcut path where we zero-out the messages
            # (for the purpose of ensuring that the training uses the messages)
            input_dict["encounter_message"] = encounter_message * 0.0
            return input_dict
        else:
            # Sample noise level
            noise = torch.round(
                torch.randn(
                    tuple(encounter_message.shape),
                    dtype=encounter_message.dtype,
                    device=encounter_message.device,
                )
                * self.noise_std
            ) * (1 / (self.num_risk_levels - 1))
            input_dict["encounter_message"] = torch.clamp(
                encounter_message + noise, 0, 1
            )
            return input_dict


class MessageDropout(Transform):
    def __init__(self, proba=0.1):
        self.proba = proba

    def apply(self, input_dict: Dict) -> Dict:
        encounter_message = input_dict["encounter_message"]
        if encounter_message.shape[0] == 0:
            # No encounter messages, so nothing to do.
            return input_dict
        assert (
            encounter_message.shape[-1] == 1
        ), "Noising is only supported for float encoded messages."
        if self.proba == 0:
            # Shortcut path where we don't add any noise
            return input_dict
        elif self.proba == -1:
            # Shortcut path where we zero-out the messages
            # (for the purpose of ensuring that the training uses the messages)
            input_dict["encounter_message"] = encounter_message * 0.0
            return input_dict
        else:
            # Sample noise level
            mask = torch.rand(
                encounter_message.shape,
                device=encounter_message.device,
                dtype=encounter_message.dtype,
            ).gt_(self.proba)
            input_dict["encounter_message"] = encounter_message * mask
            return input_dict


class FractionalEncounterDurationNoise(Transform):
    def __init__(self, fractional_noise=0.1):
        self.fractional_noise = fractional_noise

    def apply(self, input_dict: Dict) -> Dict:
        encounter_duration = input_dict["encounter_duration"]
        if encounter_duration.shape[0] == 0:
            # no encounters, nothing to do
            return input_dict
        if self.fractional_noise == -1:
            # Special codepath to remove encounter duration from the input.
            fractional_noise = 0.0
        else:
            fractional_noise = 1 + (
                torch.randn(
                    tuple(encounter_duration.shape),
                    dtype=encounter_duration.dtype,
                    device=encounter_duration.device,
                )
                * self.fractional_noise
            ).clamp_min(0)
        input_dict["encounter_duration"] = encounter_duration * fractional_noise
        return input_dict


class DropHealthHistory(Transform):
    def __init__(
        self, symptom_dropout=0.3, test_result_dropout=0.3, noise_coarseness=1
    ):
        self.symptom_dropout = symptom_dropout
        self.test_result_dropout = test_result_dropout
        self.noise_coarseness = noise_coarseness

    def apply(self, input_dict: Dict) -> Dict:
        health_history = input_dict["health_history"]
        # Get noise. Like in the other transforms, we have a codepath where
        # setting the dropout to -1 results in all symptoms being dropped.
        if self.symptom_dropout == -1 and self.test_result_dropout == -1:
            # Speedy codepath where we skip the rng calls and just multiply by 0
            input_dict["health_history"] = health_history * 0
            return input_dict
        elif self.symptom_dropout == 0 and self.test_result_dropout == 0:
            # We're not adding any noise, so nothing to do here
            return input_dict
        symptom_dropout = self.symptom_dropout if self.symptom_dropout != -1.0 else 1.0
        test_result_dropout = (
            self.test_result_dropout if self.test_result_dropout != -1.0 else 1.0
        )
        # Make a noise mask based on the `coarseness`
        if self.noise_coarseness == 0:
            # Fine noise -- meaning that if symptom A is dropped in day 1, it
            # doesn't necessarily mean that it's dropped in day 2.
            # Should simulate a scenario where the user "forgets" to enter symptoms
            # in a given day.
            symptom_mask = torch.rand(
                (health_history.shape[0], health_history.shape[-1] - 1),
                dtype=health_history.dtype,
                device=health_history.device,
            ).gt_(symptom_dropout)
            test_result_mask = torch.rand(
                (health_history.shape[0], 1),
                dtype=health_history.dtype,
                device=health_history.device,
            ).gt_(test_result_dropout)
            full_mask = torch.cat([symptom_mask, test_result_mask], dim=-1)
        elif self.noise_coarseness == 1:
            # Semi-coarse noise -- meaning that if a symptom is dropped in day 1,
            # it's guaranteed to be dropped in all the days. However, just because
            # one symptom is dropped doesn't mean that all others are dropped as well.
            # Should simulate a scenario where the user "neglects" to enter particular
            # symptoms.
            symptom_mask = torch.rand(
                (1, health_history.shape[-1] - 1),
                dtype=health_history.dtype,
                device=health_history.device,
            ).gt_(symptom_dropout)
            test_result_mask = torch.rand(
                (1, 1), dtype=health_history.dtype, device=health_history.device,
            ).gt_(test_result_dropout)
            full_mask = torch.cat([symptom_mask, test_result_mask], dim=-1)
        elif self.noise_coarseness == 2:
            # Coarse noise -- meaning that either all symptoms are dropped or
            # none of them are.
            symptom_mask = (
                torch.rand((), dtype=health_history.dtype, device=health_history.device)
                .gt_(symptom_dropout)
                .repeat(1, health_history.shape[-1] - 1)
            )
            test_result_mask = (
                torch.rand((), dtype=health_history.dtype, device=health_history.device)
                .gt_(test_result_dropout)
                .repeat(1, 1)
            )
            full_mask = torch.cat([symptom_mask, test_result_mask], dim=-1)
        else:
            raise NotImplementedError
        input_dict["health_history"] = health_history * full_mask
        return input_dict


class DropHealthProfile(Transform):
    def __init__(self, preexisting_condition_dropout=0.3):
        self.preexisting_condition_dropout = preexisting_condition_dropout

    def apply(self, input_dict: Dict) -> Dict:
        if self.preexisting_condition_dropout == 0:
            # No noise to add, so we take this superfast codepath
            return input_dict
        health_profile = input_dict["health_profile"].clone()
        pec_dropout = (
            self.preexisting_condition_dropout
            if self.preexisting_condition_dropout != -1.0
            else 1.0
        )
        pec_mask = torch.rand(
            (health_profile.shape[0] - 2,),
            dtype=health_profile.dtype,
            device=health_profile.device,
        ).gt_(pec_dropout)
        health_profile[2:] = health_profile[2:] * pec_mask
        input_dict["health_profile"] = health_profile
        return input_dict


class DigitizeInfectiousness(Transform):
    DEFAULT_BINS = np.linspace(0, 0.7, 49)
    INFINITY_BIN = 0.7

    def __init__(self, bins=None, inversion_mode="mode"):
        self.bins = (
            np.asarray(bins)
            if bins is not None
            else cu.get_infectiousness_bins(copy=True)
        )
        self.dequantization_bins = np.concatenate(
            [self.bins[0:1], 0.5 * (self.bins[1:] + self.bins[:-1])]
        )
        self.inversion_mode = inversion_mode

    def apply(self, input_dict: Dict) -> Dict:
        infectiousness_history = input_dict["infectiousness_history"]
        infectiousness_history = torch.from_numpy(
            np.digitize(infectiousness_history.numpy(), bins=self.bins, right=True)
        )
        input_dict["infectiousness_history"] = infectiousness_history
        return input_dict

    def inverse_apply(self, output_dict):
        infectiousness = output_dict["latent_variable"]
        if self.inversion_mode == "mode":
            binned_infectiousness = torch.argmax(infectiousness, dim=-1)
            dequantized_infectiousness = torch.from_numpy(
                np.take(
                    self.dequantization_bins, binned_infectiousness.numpy(), mode="clip"
                )
            )
        elif self.inversion_mode == "mean":
            with torch.no_grad():
                infectiousness = torch.softmax(infectiousness, dim=-1)
            dequant_bins = np.concatenate(
                [self.dequantization_bins, [self.INFINITY_BIN]]
            )
            dequantized_infectiousness = torch.from_numpy(
                infectiousness.numpy() * dequant_bins[None, None]
            ).sum(-1)
        elif self.inversion_mode == "none":
            with torch.no_grad():
                dequantized_infectiousness = torch.softmax(infectiousness, dim=-1)
        else:
            raise NotImplementedError
        output_dict["latent_variable"] = dequantized_infectiousness
        return output_dict


class ViralLoadToInfectiousness(Transform):
    def __init__(self, multiplier=0.5, override_vl2i=False, vl2i_clip_min=None):
        self.multiplier = multiplier
        self.override_vl2i = override_vl2i
        self.vl2i_clip_min = vl2i_clip_min

    def inverse_apply(self, output_dict):
        assert "viral_load_history" in output_dict
        if "vl2i_multiplier" in output_dict and not self.override_vl2i:
            multiplier = output_dict["vl2i_multiplier"][:, 0:1, 0:1]
            if self.vl2i_clip_min is not None:
                multiplier = multiplier.clamp_min(self.vl2i_clip_min)
        else:
            multiplier = self.multiplier
        infectiousness_history = output_dict["viral_load_history"] * multiplier
        output_dict["latent_variable"] = infectiousness_history
        output_dict["infectiousness_history"] = infectiousness_history
        return output_dict


class MuPlusAlphaSigmaInfectiousness(Transform):
    def __init__(self, alpha=0.0):
        """
        Parameters
        ----------
        alpha : float
            This is the alpha we use for predicting mu + (alpha * sigma).
        """
        self.alpha = alpha

    def inverse_apply(self, output_dict):
        assert output_dict["infectiousness_history"].shape[-1] == 2
        mu, sigma = (
            output_dict["infectiousness_history"][:, :, 0:1],
            output_dict["infectiousness_history"][:, :, 1:2],
        )
        output_dict["latent_variable"] = output_dict["infectiousness_history"] = mu + (
            self.alpha * sigma
        )
        return output_dict


class QuantilePicker(Transform):
    def __init__(self, quantile_idx=None):
        """
        Parameters
        ----------
        quantile_idx : int
            Index of the quantile to pick. If set to None, it will auto-select
            the central quantile.
        """
        self.quantile_idx = quantile_idx

    def inverse_apply(self, output_dict):
        num_quantiles = output_dict["infectiousness_history"].shape[-1]
        if self.quantile_idx is None:
            quantile_idx = num_quantiles // 2
        else:
            quantile_idx = self.quantile_idx
            assert quantile_idx < num_quantiles
        output_dict["latent_variable"] = output_dict[
            "infectiousness_history"
        ] = output_dict["infectiousness_history"][
            :, :, quantile_idx : (quantile_idx + 1)
        ]
        return output_dict


# ------------------------------
# ------- Config Parsing -------
# ------------------------------


def get_transforms(config):
    transforms = []
    for name in config.get("names", []):
        cls = globals()[name]
        kwargs = config.get("kwargs", {}).get(name, {})
        transforms.append(cls(**kwargs))
    return Compose(transforms)


def get_pre_transforms(config):
    transforms = []
    for name in config.get("names", []):
        cls = globals()[name]
        kwargs = config["kwargs"].get(name, {})
        transforms.append(cls(**kwargs))
    return ComposePreTransforms(transforms)
