import torch
import numpy as np

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

        num_tasks = len(self.continual_dataset)

        # Inizializziamo le matrici (T x T) con NaN per tracciare lo storico di ogni metrica
        metric_names = [metric.name for metric in self.metrics]
        performance_matrices = {m: np.full((num_tasks, num_tasks), np.nan) for m in metric_names}

        for task_index in range(num_tasks):

            self.continual_model.train()

            print(f"\n--- Training for task: {task_index} , {self.continual_dataset.get_task_category(task_index)} ---")

            train_dataset, eval_dataset = self.continual_dataset.get_task_data(task_index)

            self.continual_model.start_task(task_index, train_dataset, self.trainer_arguments)

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

            # Evaluate on ALL seen tasks (da 0 fino a task_index INCLUSO)
            summary_metrics = { metric_name: [] for metric_name in metric_names }
            step_logs = {} # Dizionario unico per i log di questo step

            # MODIFICA: iteriamo da 0 fino a task_index (incluso)
            for eval_task_index in range(task_index + 1):

                eval_dataset = self.continual_dataset.get_task_data_evaluation(eval_task_index)
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=self.trainer_arguments.batch_size,
                    shuffle=False,
                    num_workers=4
                )

                results = Evaluator.evaluate(self.continual_model, eval_dataloader, self.metrics, device=self.device)

                print(f"Performances on task {eval_task_index} after training on task {task_index}:")

                # Salviamo i risultati nella matrice e aggiorniamo le metriche summary e logs
                for metric_name in metric_names:
                    val = results[metric_name]

                    # Popoliamo la riga corrente (task_index) colonna (eval_task_index)
                    performance_matrices[metric_name][task_index, eval_task_index] = val

                    # Aggiorniamo summary
                    summary_metrics[metric_name].append(val)

                    # Prepariamo il dizionario dei log differenziando per task valutato
                    prefix = f"Eval_on_Task_{eval_task_index}"
                    step_logs[f"{prefix}/{metric_name}"] = val

                    print(f"  - {metric_name}: {val:.4f}")

            print(f"\nSummary metrics after training on task {task_index}:")
            for metric_name, values in summary_metrics.items():
                avg_value = sum(values) / len(values)
                summary_metrics[metric_name] = avg_value
                print(f"Average {metric_name}: {avg_value:.4f}")

                if self.logger:
                    step_logs[f"Summary_Average/{metric_name}"] = avg_value

            # Logghiamo tutto insieme per questo step di addestramento
            if self.logger:
                self.logger.log(step_logs)


        # =========================================================================
        # CALCOLO DELL'AVERAGE FORGETTING FINALE (Fine del flusso)
        # =========================================================================
        print("\n" + "="*50)
        print("CALCOLO AVERAGE FORGETTING FINALE")
        print("="*50)

        final_forgetting_logs = {}

        for metric_name in metric_names:
            matrix = performance_matrices[metric_name]
            forgetting_per_task = []

            # Calcoliamo il forgetting solo per i task fino al penultimo
            for j in range(num_tasks - 1):
                # Valore massimo ottenuto in passato sul task j
                max_past_perf = np.nanmax(matrix[j:num_tasks-1, j])

                # Valore finale al termine di tutto sul task j
                final_perf = matrix[num_tasks-1, j]

                forgetting = max_past_perf - final_perf
                forgetting_per_task.append(forgetting)

            # Media globale (esclude l'ultimo task che non ha subito forgetting)
            avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0
            print(f"Average Forgetting per {metric_name}: {avg_forgetting:.4f}")

            if self.logger:
                final_forgetting_logs[f"Final_Forgetting/{metric_name}"] = avg_forgetting

        if self.logger:
            self.logger.log(final_forgetting_logs)

        return performance_matrices
