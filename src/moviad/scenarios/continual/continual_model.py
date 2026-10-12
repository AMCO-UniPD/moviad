from abc import abstractmethod, ABC

from moviad.models.vad_model import VADModel
from moviad.datasets.vad_dataset import VADDataset
from moviad.models.training_args import TrainingArgs
from moviad.utilities.evaluation.metrics import Metric

import torch
class ContinualModel(ABC):

    def __init__(self, vad_model: VADModel):
        self.vad_model = vad_model

    def to(self, device: torch.device):
        self.vad_model.to(device)
        return self

    def train(self, mode: bool = True):
        self.vad_model.train(mode)
        self.training = mode
        return self

    def eval(self):
        self.vad_model.eval()
        self.training = False
        return self

    def forward(self, batch: torch.Tensor):
        return self.vad_model(batch)

    def __call__(self, batch: torch.Tensor):
        return self.forward(batch)

    @abstractmethod
    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None): ...

    @abstractmethod
    def train_task(self,
                   task_index: int,
                   train_dataset:VADDataset,
                   eval_dataset:VADDataset,
                   metrics:list[Metric],
                   device: torch.device,
                   logger = None,
                   train_args:TrainingArgs = None): ...

    @abstractmethod
    def end_task(self, task_index:int, train_dataset: VADDataset, train_args: TrainingArgs = None): ...

    def get_size(self):
        pass
