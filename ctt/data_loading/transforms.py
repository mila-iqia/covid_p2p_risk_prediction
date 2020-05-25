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


class FractionalEncounterDurationNoise(Transform):
    def __init__(self, fractional_noise=0.1):
        self.fractional_noise = fractional_noise

    def apply(self, input_dict: Dict) -> Dict:
        encounter_duration = input_dict["encounter_duration"]
        if self.fractional_noise == -1:
            # Special codepath to remove encounter duration from the input.
            fractional_noise = 0.
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
