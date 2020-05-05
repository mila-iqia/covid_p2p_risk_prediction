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
        # Sample noise level
        noise = torch.round(
            torch.randn(
                tuple(encounter_message.shape),
                dtype=encounter_message.dtype,
                device=encounter_message.device,
            )
            * self.noise_std
        ) * (1 / (self.num_risk_levels - 1))
        input_dict["encounter_message"] = torch.clamp(encounter_message + noise, 0, 1)
        return input_dict


class FractionalEncounterDurationNoise(Transform):
    def __init__(self, fractional_noise=0.1):
        self.fractional_noise = fractional_noise

    def apply(self, input_dict: Dict) -> Dict:
        encounter_duration = input_dict["encounter_duration"]
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