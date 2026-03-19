import torch

from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import Metric

class ContinualTrainer:

    def __init__(self, 
                 continual_dataset: ContinualDataset, 
                 continual_model: ContinualModel, 
                 device: torch.device, 
                 metrics: list[Metric], 
                 training_args: TrainingArgs,
                 logger: any = None
            ):
        """
        Args:
            continual_dataset (ContinualDataset): continual dataset to be used for training
            model (nn.Module): model to be trained
            trainer_arguments (TrainerArguments): arguments for the trainer
        """
        self.continual_dataset = continual_dataset
        self.continual_model = continual_model
        self.trainer_arguments = training_args
        self.metrics = metrics
        self.device = device
        self.logger = logger

    
    def train(self):

        for task_index in range(len(self.continual_dataset)):
            print(f"Training for task: {task_index} , {self.continual_dataset.get_task_category(task_index)}")

            train_dataset, eval_dataset = self.continual_dataset.get_task_data(task_index)

            self.continual_model.start_task()

            self.continual_model.train_task(
                task_index=task_index,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                train_args=self.trainer_arguments,
                metrics=self.metrics,
                device=self.device,
                logger=self.logger,
            )

            self.continual_model.end_task(task_index, train_dataset, self.trainer_arguments)

            # Evaluate on all previous tasks and get summary metrics
            summary_metrics = { metric.name: [] for metric in self.metrics }

            previous_tasks_index = self.continual_dataset.get_previous_tasks(task_index)

            for prev_task_index in previous_tasks_index:
                eval_dataset = self.continual_dataset.get_task_data_evaluation(prev_task_index)
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=self.trainer_arguments.batch_size,
                    shuffle=False,
                    num_workers=4
                )
                results = Evaluator.evaluate(self.continual_model, eval_dataloader, self.metrics, device=self.device)

                # update summary metrics
                for metric in summary_metrics.keys():
                    summary_metrics[metric].append(results[metric])

                # log metrics for the current task
                if self.logger:
                    self.logger.log(
                        {
                            f"Task_T{task_index}/eval/{metric}": results[metric] for metric in results.keys()
                        }
                    )

            print(f"Summary metrics after training on task {task_index}:")
            for metric_name, values in summary_metrics.items():
                avg_value = sum(values) / len(values)
                summary_metrics[metric_name] = avg_value
                print(f"Average {metric_name}: {avg_value}")

            if self.logger:
                self.logger.log(
                    {
                        f"Summary/{metric}": summary_metrics[metric] for metric in summary_metrics.keys()
                    }
                )

            

            

