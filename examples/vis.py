# Libs >>>
import matplotlib.pyplot as plt
import torch

from custom_onecyclelr import scheduler


# Utils >>>
def get_dummy_optimizer() -> torch.optim.Optimizer:
    # Init the optimizer
    return torch.optim.SGD(
        params=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(3, 4))]),
        lr=1e-3,
        momentum=0.9,
    )


# Vitalization >>>
def visualize_lr_schedule():
    # Parameters for the scheduler
    warmup_iters = 6
    lr_idling_iters = 8
    annealing_iters = 56
    decay_iters = 100
    max_lr = 0.01
    annealing_lr_min = 0.001
    decay_lr_min = 0.0001
    warmup_start_lr = 0.0001

    # Total iterations
    total_iters = warmup_iters + lr_idling_iters + annealing_iters + decay_iters

    # Create optimizer and scheduler
    optimizer = get_dummy_optimizer()
    lr_scheduler = scheduler.OneCycleLr(
        optimizer=optimizer,
        warmup_iters=warmup_iters,
        lr_idling_iters=lr_idling_iters,
        annealing_iters=annealing_iters,
        decay_iters=decay_iters,
        max_lr=max_lr,
        annealing_lr_min=annealing_lr_min,
        decay_lr_min=decay_lr_min,
        warmup_start_lr=warmup_start_lr,
        warmup_type="exp",
    )

    # Collect learning rates
    lrs = []
    for i in range(total_iters + 30):  # Add some extra iterations to see final behavior
        optimizer.step()
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr())

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(lrs, linewidth=2)

    # Add vertical lines to separate phases
    plt.axvline(x=warmup_iters -1, color="r", linestyle="--", alpha=0.5)
    plt.axvline(x=warmup_iters + lr_idling_iters - 1, color="r", linestyle="--", alpha=0.5)
    plt.axvline(
        x=warmup_iters + lr_idling_iters + annealing_iters - 1,
        color="r",
        linestyle="--",
        alpha=0.5,
    )
    plt.axvline(x=total_iters - 1, color="r", linestyle="--", alpha=0.5)

    # Add text labels for each phase
    plt.text(warmup_iters / 2, max_lr / 2, "Warmup", ha="center")
    plt.text(warmup_iters + lr_idling_iters / 2, max_lr * 1.01, "Idling", ha="center")
    plt.text(
        warmup_iters + lr_idling_iters + annealing_iters / 2,
        max_lr / 2,
        "Annealing",
        ha="center",
    )
    plt.text(
        warmup_iters + lr_idling_iters + annealing_iters + decay_iters / 2,
        decay_lr_min * 5,
        "Decay",
        ha="center",
    )

    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("OneCycleLR Scheduler Visualization", fontsize=14)

    # Save the figure
    plt.savefig("doc/vis/onecycle_lr_schedule.png", dpi=300, bbox_inches="tight")
    plt.show()


# Start >>>
if __name__ == "__main__":
    visualize_lr_schedule()
