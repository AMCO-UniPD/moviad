from collections.abc import Callable
import os

import torch

from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric
from moviad.trainers.trainer import Trainer

class SingleModel(ContinualModel):

    def __init__(self, model: VADModel, save_directory_path: str = None, saving_criteria: Callable = None):
        super().__init__(model)
        self.save_directory_path = save_directory_path
        self.saving_criteria = saving_criteria

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs):
        self.vad_model.reset_model()

    def train_task(self, 
                   task_index: int, 
                   train_dataset:VADDataset, 
                   eval_dataset:VADDataset,
                   metrics:list[Metric], 
                   device: torch.device, 
                   logger = None,
                   train_args:TrainingArgs = None):

        trainer = Trainer(
            train_args,
            self.vad_model,
            train_dataset,
            eval_dataset,
            metrics=metrics,
            device=device,
            logger=logger,
            logging_prefix=f"Task_T{task_index}/",
            saving_criteria=self.saving_criteria,
            save_path=os.path.join(self.save_directory_path, f"task_{task_index}.pth") if self.save_directory_path else None,
        )

        trainer.train()
        

    def end_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs):
        pass
