from __future__ import annotations
import torch
from typing import Any, Callable

from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.models import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric

class Trainer:

    def __init__(
        self,
        train_args: TrainingArgs,
        model: VADModel,
        train_dataset: VADDataset,
        eval_dataset: VADDataset | None,
        metrics: list[Metric],
        device: torch.device,
        logger: Any | None = None,
        logging_prefix: str = "",
        save_path: str | None = None,
        saving_criteria: Callable | None = None,
    ):
        self.model = model
        
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4   
        )
        
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=4
        ) if eval_dataset is not None else None

        self.device = device
        self.logger = logger
        self.logging_prefix = logging_prefix
        self.save_path = save_path
        self.saving_criteria = saving_criteria
        self.train_args = train_args
        self.metrics = metrics

    @staticmethod
    def update_best_metrics(best_metrics, metrics):
        for m in metrics.keys():
            best_metrics[m] = max(best_metrics[m], metrics[m])

        return best_metrics

    @staticmethod
    def print_metrics(metrics):
        print("\n".join([f"{k}: {v}" for k, v in metrics.items()]))

    def save_model(self, best_metrics, results):
        if self.saving_criteria and self.saving_criteria(best_metrics, results) and self.save_path is not None:
            print("Saving model...")
            torch.save(self.model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")

    def train(self):

        self.train_args.init_train(self.model)

        if self.logger:
            self.logger.config.update(self.train_args.__dict__)
            
        best_metrics = {metric.name: 0.0 for metric in self.metrics}

        for epoch in range(self.train_args.epochs):

            self.model.train()

            print(f"EPOCH: {epoch}")

            avg_batch_loss = self.model.train_epoch(epoch, self.train_dataloader, self.train_args)

            if self.logger:
                self.logger.log({
                    f"{self.logging_prefix}train/epoch" : epoch,
                    f"{self.logging_prefix}train/train_loss" : avg_batch_loss
                })

            if (epoch + 1) % self.train_args.evaluation_epoch_interval == 0:
                print("Evaluating model...")
                results = Evaluator.evaluate(self.model, self.eval_dataloader, self.metrics, self.device)

                # save the model if needed
                self.save_model(best_metrics, results)

                # update the best metrics
                best_metrics = Trainer.update_best_metrics(best_metrics, results)

                print("Training performances:")
                Trainer.print_metrics(results)

                
                if self.logger is not None:
                    if self.logging_prefix is not None:
                        self.logger.log({
                            f"{self.logging_prefix}train/{metric_name}": value for metric_name, value in results.items()
                        })