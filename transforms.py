import torch
from addict import Dict


class Transform(object):
    def apply(self, input_dict: Dict) -> Dict:
        raise NotImplementedError

    def __call__(self, input_dict: Dict) -> Dict:
        input_dict = Dict(input_dict)
        return self.apply(input_dict)


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
        ) * (1 / self.num_risk_levels)
        input_dict["encounter_message"] = torch.clamp(encounter_message + noise, 0, 1)
        return input_dict


def get_transforms(config):
    pass
