"""
UniAD Learning Rate Scheduler.
    scheduler: StepLR
    step_size: 800
    gamma: 0.1
"""

import torch


def build_scheduler(optimizer, step_size=800, gamma=0.1):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )
