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
        annealing_iters: int,
        decay_iters: int,
        annealing_lr_min: float,
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
        # Calculate the lr based on the warmup type
        match warmup_type:
            case "linear":
                lr = warmup_start_lr + (
                    (warmup_max_lr - warmup_start_lr) * (step / warmup_duration)
                )

            case "exp":
                lr = warmup_start_lr * math.pow(
                    math.pow(warmup_max_lr / warmup_start_lr, 1 / warmup_duration), step
                )

        return lr

    def _annealing_phase(
        self,
        step: int,
        annealing_duration: int,
        annealing_start_lr: float,
        annealing_min_lr: float,
    ) -> float:
        # Interpolate between start_lr and min_lr using a cosine factor
        return (
            annealing_min_lr
            + (annealing_start_lr - annealing_min_lr)
            * (1 + math.cos(math.pi * (step / annealing_duration)))
            / 2
        )

    def _decay_phase(
        self,
        step: int,
        decay_duration: int,
        decay_start_lr: float,
        decay_min_lr: float,
    ) -> float:
        # Linear decay from start_lr to min_lr
        return decay_start_lr - (
            (step / decay_duration) * (decay_start_lr - decay_min_lr)
        )

    def get_lr(self):
        return 0.01
