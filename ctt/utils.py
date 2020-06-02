from typing import Sequence
import torch


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return type(x)({key: to_device(val, device) for key, val in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([to_device(item, device) for item in x])
    elif isinstance(x, torch.nn.Module):
        return x.to(device)
    else:
        raise NotImplementedError


def momentum_accumulator(momentum):
    def _accumulator(old, new):
        return momentum * old + (1 - momentum) * new

    return _accumulator


def thermometer_encoding(x: torch.Tensor, value_range: Sequence[int], size: int):
    assert x.shape[-1] == 1
    # Make linspace and expand it to shape (1, ..., 1, size), with trailing n-1
    # singleton dimensions, where x.ndim = n.
    expanded_linspace = torch.linspace(
        value_range[0], value_range[1], size, dtype=x.dtype, device=x.device
    ).expand(*([1] * (x.dim() - 1) + [size]))
    return torch.gt(x, expanded_linspace).float()


def typed_sum_pool(x: torch.Tensor, type_: torch.Tensor, reference_types: torch.Tensor):
    # x.shape = BMC, type_.shape = BM, reference_types.shape = BT
    # Validate shapes
    assert x.ndim == 3
    if type_.ndim == 3:
        assert type_.shape[-1] == 1
        type_ = type_[..., 0]
        assert type_.shape[1] == x.shape[1]
    if reference_types.ndim == 3:
        assert reference_types.shape[-1] == 1
        reference_types = reference_types[..., 0]
    # Get a mask of shape BMT, which is 1 if entity idx m is of type index t,
    # and 0 otherwise
    type_mask = torch.eq(type_[:, :, None], reference_types[:, None, :]).float()
    # For a given type t, sum over all entities m that are of type t.
    pooled = torch.einsum("bmc,bmt->btc", x, type_mask)
    return pooled


def compute_moments(x, dim=0, num_moments=2, mask=None, eps=1e-8):
    if x.shape[dim] == 0:
        # Special codepath that returns 0s instead of nans
        shape = list(x.shape)
        shape[dim] = num_moments
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    else:
        if mask is None:
            mean = x.mean(dim=dim, keepdim=True)
        else:
            mean = (x * mask).sum(dim=dim, keepdim=True) / (
                mask.sum(dim=dim, keepdim=True) + eps
            )
        residuals = x - mean
        moments = [mean]
        for moment in range(1, num_moments):
            if mask is None:
                moments.append(residuals.pow(moment + 1).mean(dim=dim, keepdim=True))
            else:
                moments.append(
                    (residuals.pow(moment + 1) * mask).sum(dim=dim, keepdim=True)
                    / (mask.sum(dim=dim, keepdim=True) + eps)
                )
        moments = torch.cat(moments, dim=dim)
        return moments


class Symptoms:
    MODERATE = 0
    MILD = 1
    SEVERE = 2
    EXTREMELY_SEVERE = 3
    FEVER = 4
    CHILLS = 5
    GASTRO = 6
    DIARRHEA = 7
    NAUSEA_VOMITING = 8
    FATIGUE = 9
    UNUSUAL = 10
    HARD_TIME_WAKING_UP = 11
    HEADACHE = 12
    CONFUSED = 13
    LOST_CONSCIOUSNESS = 14
    TROUBLE_BREATHING = 15
    SNEEZING = 16
    COUGH = 17
    RUNNY_NOSE = 18
    ACHES = 19
    SORE_THROAT = 20
    SEVERE_CHEST_PAIN = 21
    LOSS_OF_TASTE = 22
    MILD_TROUBLE_BREATHING = 23
    LIGHT_TROUBLE_BREATHING = 24
    MODERATE_TROUBLE_BREATHING = 25
    HEAVY_TROUBLE_BREATHING = 26

    DROP_IN_GROUPS = [
        [MILD],
        [MODERATE],
        [SEVERE],
        [EXTREMELY_SEVERE],
        [FEVER],
        [FEVER, MODERATE],
        [FEVER, CHILLS],
        [GASTRO],
        [GASTRO, DIARRHEA],
        [GASTRO, DIARRHEA, NAUSEA_VOMITING],
        [GASTRO, NAUSEA_VOMITING],
        [FATIGUE],
        [FATIGUE, UNUSUAL],
        [FATIGUE, LOST_CONSCIOUSNESS],
        [FATIGUE, HARD_TIME_WAKING_UP],
        [FATIGUE, HEADACHE],
        [FATIGUE, CONFUSED],
        [TROUBLE_BREATHING],
        [TROUBLE_BREATHING, SEVERE_CHEST_PAIN],
        [TROUBLE_BREATHING, SNEEZING],
        [TROUBLE_BREATHING, COUGH],
        [TROUBLE_BREATHING, RUNNY_NOSE],
        [TROUBLE_BREATHING, SORE_THROAT],
        [LOSS_OF_TASTE],
        [GASTRO, MODERATE],
        [FATIGUE, GASTRO],
        [MILD, MILD_TROUBLE_BREATHING, MILD_TROUBLE_BREATHING, LIGHT_TROUBLE_BREATHING],
        [MODERATE, TROUBLE_BREATHING, MODERATE_TROUBLE_BREATHING],
        [SEVERE, TROUBLE_BREATHING, HEAVY_TROUBLE_BREATHING],
        [EXTREMELY_SEVERE, TROUBLE_BREATHING, HEAVY_TROUBLE_BREATHING],
    ]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
