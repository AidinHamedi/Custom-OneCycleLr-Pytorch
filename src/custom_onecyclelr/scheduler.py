# Libs >>>
import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step


# Core >>>
class OneCycleLr(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        lr_idling_iters: int,
        annealing_iters: int,
        decay_iters: int,
        max_lr: float,
        annealing_lr_min: float,
        decay_lr_min: float,
        warmup_start_lr: float = 0.001,
        warmup_type: Literal["linear", "exp"] = "exp",
        last_epoch: int = -1,
        verbose="deprecated",
    ) -> None:
        # Init the super class
        super().__init__(optimizer, last_epoch, verbose)

        # Init the attributes
        self.warmup_iters = warmup_iters
        self.lr_idling_iters = lr_idling_iters
        self.annealing_iters = annealing_iters
        self.decay_iters = decay_iters
        self.max_lr = max_lr
        self.annealing_lr_min = annealing_lr_min
        self.decay_lr_min = decay_lr_min
        self.warmup_start_lr = warmup_start_lr
        self.warmup_type = warmup_type

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
        """Retrieve the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        # Calculate the lr based on the current phase
        if self.last_epoch <= self.warmup_iters:  # Warmup phase
            lr = self._warmup_phase(
                step=self.last_epoch,
                warmup_duration=self.warmup_iters,
                warmup_start_lr=self.warmup_start_lr,
                warmup_max_lr=self.max_lr,
                warmup_type=self.warmup_type,
            )
        elif self.last_epoch <= (
            self.warmup_iters + self.lr_idling_iters
        ):  # LR idling phase
            lr = self.max_lr
        elif self.last_epoch <= (
            self.warmup_iters + self.lr_idling_iters + self.annealing_iters
        ):  # Annealing phase
            lr = self._annealing_phase(
                step=self.last_epoch - (self.warmup_iters + self.lr_idling_iters),
                annealing_duration=self.annealing_iters,
                annealing_start_lr=self.max_lr,
                annealing_min_lr=self.annealing_lr_min,
            )
        elif self.last_epoch <= (
            self.warmup_iters
            + self.lr_idling_iters
            + self.annealing_iters
            + self.decay_iters
        ):  # Decay phase
            lr = self._decay_phase(
                step=self.last_epoch
                - (self.warmup_iters + self.lr_idling_iters + self.annealing_iters),
                decay_duration=self.decay_iters,
                decay_start_lr=self.annealing_lr_min,
                decay_min_lr=self.decay_lr_min,
            )
        else:  # Min lr
            lr = self.decay_lr_min

        # Return the lr
        return [lr for _ in self.optimizer.param_groups]
