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
