from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm

from moviad.models.training_args import TrainingArgs

class VADModel(nn.Module):

    def train_epoch(
        self, epoch, train_dataloader, training_args: TrainingArgs
    ):
        avg_batch_loss = 0

        # train the model
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)

        if training_args.scheduler is not None:
            training_args.scheduler.step()

        return avg_batch_loss

    @abstractmethod
    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs): ...

    @abstractmethod
    def train_chunk(self, train_dataloader: torch.utils.data.DataLoader, training_args: TrainingArgs):
        pass

    def reset_model(self):
        pass

    def save(self, save_path: str):
        pass

    def get_model_size(self):
        pass