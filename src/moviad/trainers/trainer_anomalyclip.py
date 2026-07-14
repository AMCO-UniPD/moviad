import torch
from typing import Any, Callable

from moviad.trainers.trainer import Trainer
from moviad.models.training_args import TrainingArgs
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric
from moviad.utilities.evaluation.evaluator import Evaluator


class AnomalyCLIPTrainer(Trainer):
    
    def __init__(
        self,
        train_args: TrainingArgs,
        model,  # AnomalyCLIPModel
        train_dataset: VADDataset,
        eval_dataset: VADDataset | None,
        metrics: list[Metric],
        device: torch.device,
        logger: Any | None = None,
        logging_prefix: str = "",
        save_path: str | None = None,
        saving_criteria: Callable | None = None,
    ):
        # Initialize optimizer before calling parent constructor
        # Only optimize prompt learner parameters

        self.optimizer = train_args.optimizer
        
        # Call parent constructor
        super().__init__(
            train_args=train_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metrics=metrics,
            device=device,
            logger=logger,
            logging_prefix=logging_prefix,
            save_path=save_path,
            saving_criteria=saving_criteria
        )
    
    def train(self):
        
        self.train_args.init_train(self.model)
        
        if self.logger:
            self.logger.config.update(self.train_args.__to_dict__())
        
        best_metrics = {metric.name: 0.0 for metric in self.metrics}
        
        for epoch in range(self.train_args.epochs):

            # Keep CLIP model frozen, only train prompt learner
            self.model.model.eval()
            self.model.prompt_learner.train()
            
            print(f"EPOCH: {epoch}")
            
            # Training epoch
            avg_batch_loss = self.model.train_epoch(
                epoch, 
                self.train_dataloader, 
                self.train_args
            )
     
            
            # Log training loss
            if self.logger:
                self.logger.log({
                    f"{self.logging_prefix}train/epoch": epoch,
                    f"{self.logging_prefix}train/train_loss": avg_batch_loss
                })
            
            # Evaluation
            if (epoch + 1) % self.train_args.evaluation_epoch_interval == 0:
                print("Evaluating model...")
                
                # Set to eval mode for evaluation
                self.model.model.eval()
                self.model.prompt_learner.eval()
                
                results = Evaluator.evaluate(
                    self.model, 
                    self.eval_dataloader, 
                    self.metrics, 
                    self.device
                )
                
                # Save model if needed
                self.save_model(best_metrics, results)
                
                # Update best metrics
                best_metrics = Trainer.update_best_metrics(best_metrics, results)
                
                print("Training performances:")
                Trainer.print_metrics(results)
                
                # Log evaluation metrics
                if self.logger is not None:
                    if self.logging_prefix is not None:
                        self.logger.log({
                            f"{self.logging_prefix}/eval/{metric_name}": value 
                            for metric_name, value in results.items()
                        })
        
        # Final save if no criteria specified
        if self.saving_criteria is None and self.save_path is not None:
            print("Saving final model...")
            self.model.save_model(self.save_path)
            print(f"Model saved to {self.save_path}")
