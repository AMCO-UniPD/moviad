import torch

from moviad.datasets.vad_dataset import VADDataset
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.models.cfa.cfa import CFA
from moviad.utilities.evaluation.metrics import Metric

class CFAContinual(Replay):

    def __init__(self, cfa_model:CFA, fix_centroids:bool = False, memory_size: int = 300, replay_ratio=0.5):
        super().__init__(cfa_model, memory_size, replay_ratio)
        self.fix_centroids = fix_centroids

    def update_memory_bank(self, task_index: int, train_dataloader: torch.utils.data.DataLoader):
        task_memory_bank = self.vad_model.initialize_memory_bank(train_dataloader)
        new_memory_bank = self.vad_model.memory_bank * (task_index / (task_index + 1)) + task_memory_bank / (task_index + 1)
        self.vad_model.memory_bank = torch.nn.Parameter(new_memory_bank, requires_grad=False)

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=train_args.num_workers,
        )

        # initialize the memory bank for CFA or update it with the new task data
        if task_index == 0:
            self.vad_model.memory_bank = torch.nn.Parameter(self.vad_model.initialize_memory_bank(train_dataloader), requires_grad=False)
        elif not self.fix_centroids:
            self.update_memory_bank(task_index, train_dataloader)
