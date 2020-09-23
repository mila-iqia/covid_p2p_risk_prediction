import torch
import numpy as np
from typing import Dict, Callable, Union
import time


class BinaryRejectionSampler(object):
    def __init__(
        self,
        rejection_criterion: Union[str, Callable],
        rejection_weight: float,
        seed: float = None,
    ):
        if isinstance(rejection_criterion, str):
            rejection_criterion = globals().get(rejection_criterion, None)
            assert (
                rejection_criterion is not None
            ), f"Unknown rejection criterion: {rejection_criterion}."
        self.rejection_criterion = rejection_criterion
        self.rejection_weight = rejection_weight
        # DOUBLE DANGER: Make sure seeds are not only different for every worker,
        # but also for every epoch.
        self.rng = np.random.RandomState(
            seed or (int(time.time() * 10000000) % 4294967295)
        )

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, sample: Dict) -> Union[Dict, None]:
        # The rejection_criterion outputs whether the sample qualifies for a rejection.
        sample_is_rejectable = self.rejection_criterion(sample)
        # Let's play out what happens here.
        # Say rejection weight is 1. This means that coin < 1, and we're essentially
        # gonna return all samples, rejectable or not. This means that all samples must
        # be weighted by 1. Now, say the rejection weight is close to 0. This means
        # we're only going to return the non-rejected samples, but with a weight of 0.
        if sample_is_rejectable:
            # Sample qualifies for rejection
            coin = self.rng.rand()
            if coin < self.rejection_weight:
                sample["sample_weight"] = torch.tensor([1.0])
                return sample
            else:
                return None
        else:
            # Sample does not qualify for rejection
            # Prepare to return it, but with a down-weight
            sample["sample_weight"] = torch.tensor([self.rejection_weight])
            return sample


def reject_nonzero_infectiousness(sample):
    return not bool(sample["infectiousness_history"].max().gt(0))
