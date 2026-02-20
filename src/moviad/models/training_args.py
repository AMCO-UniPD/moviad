from dataclasses import dataclass
from typing import Callable
import torch

@dataclass
class TrainingArgs: 
    batch_size: int
    epochs: int
    lr: float
    evaluation_epoch_interval: int = 1 
    optimizer: torch.optim.Optimizer | None = None
    loss_function: Callable | None = None

    def __to_dict__(self):

        optimizer_dict = self.optimizer_to_dict(self.optimizer) if self.optimizer else None

        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "evaluation_epoch_interval": self.evaluation_epoch_interval,
            "loss_function": self.loss_function,
            "optimizer": optimizer_dict,
        }
    
    @staticmethod
    def optimizer_to_dict(optimizer: torch.optim.Optimizer):
        return {
            "type": optimizer.__class__.__name__,
            "param_groups": [
                {
                    "lr": group.get("lr"),
                    "weight_decay": group.get("weight_decay"),
                    "momentum": group.get("momentum"),
                    "betas": group.get("betas"),
                    "eps": group.get("eps"),
                    "amsgrad": group.get("amsgrad"),
                    "nesterov": group.get("nesterov"),
                }
                for group in optimizer.param_groups
            ],
        }

    def init_train(self, model: torch.nn.Module):
        pass