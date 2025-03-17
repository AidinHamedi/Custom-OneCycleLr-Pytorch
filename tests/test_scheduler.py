# Libs >>>
import pytest

# Module >>>
from custom_onecyclelr import scheduler

from .utils import get_dummy_optimizer


# Tests >>>
def test_warmup_phase_linear():
    # Test warmup type: linear
    assert scheduler.OneCycleLr._warmup_phase(  # last step retuning max lr
        None,
        step=10,
        warmup_duration=10,
        warmup_start_lr=0.001,
        warmup_max_lr=0.01,
        warmup_type="linear",
    ) == pytest.approx(0.01, abs=1e-8)

    assert scheduler.OneCycleLr._warmup_phase(  # first step retuning min lr
        None,
        step=0,
        warmup_duration=10,
        warmup_start_lr=0.001,
        warmup_max_lr=0.01,
        warmup_type="linear",
    ) == pytest.approx(0.001, abs=1e-8)

    assert (
        scheduler.OneCycleLr._warmup_phase(  # 0.5 of the duration returns the mean of start and max lr
            None,
            step=5,
            warmup_duration=10,
            warmup_start_lr=0.001,
            warmup_max_lr=0.01,
            warmup_type="linear",
        )
        == pytest.approx(0.0055, abs=1e-8)
    )


def test_warmup_phase_exp():
    # Test warmup type: exp
    assert scheduler.OneCycleLr._warmup_phase(  # last step retuning max lr
        None,
        step=10,
        warmup_duration=10,
        warmup_start_lr=0.001,
        warmup_max_lr=0.01,
        warmup_type="exp",
    ) == pytest.approx(0.01, abs=1e-8)

    assert scheduler.OneCycleLr._warmup_phase(  # first step retuning min lr
        None,
        step=0,
        warmup_duration=10,
        warmup_start_lr=0.001,
        warmup_max_lr=0.01,
        warmup_type="exp",
    ) == pytest.approx(0.001, abs=1e-8)

    assert scheduler.OneCycleLr._warmup_phase( # 0.5 of the duration returns the mean of start and max lr
        None,
        step=5,
        warmup_duration=10,
        warmup_start_lr=0.001,
        warmup_max_lr=0.01,
        warmup_type="exp",
    ) == pytest.approx(0.00316227, abs=1e-8)
