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
