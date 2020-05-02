import torch
import math


class WarmCosineMixin(object):
    """A Mixin class for torch.optim.Optimizer that implements a warmup + cosine annealing schedule."""
    def __init__(self, *args, num_warmup_steps, num_steps, eta_min, eta_max, **kwargs):
        if "lr" not in kwargs:
            kwargs["lr"] = 0.0
        super().__init__(*args, **kwargs)
        self.num_warmup_steps = num_warmup_steps
        self.num_steps = num_steps
        self.eta_min = eta_min
        self.eta_max = eta_max
        # Privates
        self._step = 0
        self._set_lr()

    def _set_lr(self):
        rate = self.rate()
        # noinspection PyUnresolvedReferences
        for group in self.param_groups:
            group["lr"] = rate

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        self._set_lr()
        # noinspection PyUnresolvedReferences
        super().step(closure=closure)

    def rate(self, step=None):
        """Return a learning rate at a step (given or taken from internal attribute)."""
        if step is None:
            step = self._step
        if step == 0:
            return 0.0
        if step > self.num_steps:
            return self.eta_min
        cos_lr = (
            self.eta_min
            + (self.eta_max - self.eta_min)
            * (1 + math.cos(math.pi * step / self.num_steps))
            / 2
        )
        lin_lr = (step / self.num_warmup_steps) * self.eta_max
        return min(lin_lr, cos_lr)


class WarmCosineAdam(WarmCosineMixin, torch.optim.Adam):
    pass


# noinspection PyUnresolvedReferences
class WarmCosineRMSprop(WarmCosineMixin, torch.optim.RMSprop):
    pass


def __getattr__(name):
    obj_in_globals = globals().get(name, None)
    if obj_in_globals is not None:
        assert issubclass(obj_in_globals, torch.optim.Optimizer)
        return obj_in_globals
    # Object not found in globals, look for optim
    return getattr(torch.optim, name)
