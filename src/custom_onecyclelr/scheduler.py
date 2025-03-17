# Libs >>>
import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


# Core >>>
class OneCycleLr(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        lr_idling_iters: int,
        sec_phase_iters: int,
        decay_iters: int,
        sec_phase_lr_min: float,
        decay_lr_min: float,
        warmup_start_lr: float = 0.001,
        warmup_type: Literal["linear", "exp"] = "exp",
        last_epoch: int = -1,
        verbose="deprecated",
    ) -> None:
        # Init the super class
        super().__init__(optimizer, last_epoch, verbose)

    def _warmup_phase(
        self,
        step: int,
        warmup_duration: int,
        warmup_start_lr: float,
        warmup_max_lr: float,
        warmup_type: Literal["linear", "exp"] = "exp",
    ) -> float:
        pass
