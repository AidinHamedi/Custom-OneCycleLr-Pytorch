# Libs >>>
import torch


# Utils >>>
def get_dummy_optimizer() -> torch.optim.Optimizer:
    # Init the optimizer
    return torch.optim.SGD(
        params=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4))]),
        lr=1e-3,
        momentum=0.9,
    )
