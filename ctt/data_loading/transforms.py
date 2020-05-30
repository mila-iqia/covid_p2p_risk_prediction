import numpy as np

import torch
from torchvision.transforms import Compose

from addict import Dict


# ------------------------------
# ------- Infrastructure -------
# ------------------------------


class Transform(object):
    def apply(self, input_dict: Dict) -> Dict:
        raise NotImplementedError

    def __call__(self, input_dict: Dict) -> Dict:
        input_dict = Dict(input_dict)
        return self.apply(input_dict)


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


class BinomialMessageNoise(Transform):
    pass


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
            # Speedy codepath where we skip the rng calls
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


# ------------------------------
# ------- Pre-Transforms -------
# ------------------------------


class EncounterDropout(PreTransform):
    def __init__(self, dropout_proba=0.5):
        self.dropout_proba = dropout_proba

    def apply(
        self, human_day_info: dict, human_idx: int = None, day_idx: int = None
    ) -> dict:
        # TODO
        pass


class ClusteringNoise(PreTransform):
    def __init__(self, expansion_factor=1.5):
        pass

    def apply(
        self, human_day_info: dict, human_idx: int = None, day_idx: int = None
    ) -> dict:
        encounter_info = human_day_info["observed"]["candidate_encounters"]
        encounter_is_contagion = human_day_info["unobserved"]["exposure_encounter"]
        # The plan is to expand the encounters.
        # For starters, we sample which encounters we want to replicate.
        replication_mask = np.random.binomial(
            n=1, p=0.5, size=(encounter_info.shape[0],)
        ).astype("bool")
        encounter_info_originals = encounter_info[replication_mask, :]
        encounter_is_contagion_originals = encounter_is_contagion[replication_mask, :]
        # TODO Continue
        raise NotImplementedError


# ------------------------------
# ------- Config Parsing -------
# ------------------------------


def get_transforms(config):
    transforms = []
    for name in config.get("names", []):
        cls = globals()[name]
        kwargs = config["kwargs"].get(name, {})
        transforms.append(cls(**kwargs))
    return Compose(transforms)


def get_pre_transforms(config):
    transforms = []
    for name in config.get("names", []):
        cls = globals()[name]
        kwargs = config["kwargs"].get(name, {})
        transforms.append(cls(**kwargs))
    return ComposePreTransforms(transforms)
