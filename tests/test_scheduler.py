# Libs >>>
import pytest

# Module >>>
from custom_onecyclelr import scheduler

from .utils import get_dummy_optimizer


# Tests >>>
@pytest.mark.parametrize(
    "warmup_type,step,expected_lr",
    [
        # Linear warmup tests
        ("linear", 0, 0.001),  # first step returning min lr
        (
            "linear",
            5,
            0.0055,
        ),  # 0.5 of the duration returns the mean of start and max lr
        ("linear", 10, 0.01),  # last step returning max lr
        # Exponential warmup tests
        ("exp", 0, 0.001),  # first step returning min lr
        ("exp", 5, 0.00316227),  # 0.5 of the duration
        ("exp", 10, 0.01),  # last step returning max lr
    ],
)
def test_warmup_phase(warmup_type, step, expected_lr):
    lr = scheduler.OneCycleLr._warmup_phase(
        None,
        step=step,
        warmup_duration=10,
        warmup_start_lr=0.001,
        warmup_max_lr=0.01,
        warmup_type=warmup_type,
    )
    assert lr == pytest.approx(expected_lr, abs=1e-8)


@pytest.mark.parametrize(
    "step,expected_lr",
    [
        # cosine annealing tests
        (0, 0.01),  # first step returning max lr
        (5, 0.0055),  # 0.5 of the duration
        (10, 0.001),  # last step returning min lr
    ],
)
def test_annealing_phase(step, expected_lr):
    lr = scheduler.OneCycleLr._annealing_phase(
        None,
        step=step,
        annealing_duration=10,
        annealing_start_lr=0.01,
        annealing_min_lr=0.001,
    )
    assert lr == pytest.approx(expected_lr, abs=1e-8)


@pytest.mark.parametrize(
    "step,expected_lr",
    [
        # cosine annealing tests
        (0, 0.01),  # first step returning max lr
        (5, 0.0055),  # 0.5 of the duration
        (10, 0.001),  # last step returning min lr
    ],
)
def test_decay_phase(step, expected_lr):
    lr = scheduler.OneCycleLr._decay_phase(
        None,
        step=step,
        decay_duration=10,
        decay_start_lr=0.01,
        decay_min_lr=0.001,
    )
    assert lr == pytest.approx(expected_lr, abs=1e-8)
