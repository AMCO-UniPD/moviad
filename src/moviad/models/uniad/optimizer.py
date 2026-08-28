"""
UniAD Optimizer.
    optimizer: AdamW
    lr: 0.0001
    betas: [0.9, 0.999]
    weight_decay: 0.0001
"""

import torch


def build_optimizer(model, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4):
    return torch.optim.AdamW(
        model.uniad_core.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )