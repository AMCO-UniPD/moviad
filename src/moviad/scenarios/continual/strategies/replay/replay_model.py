import torch
from tqdm import trange
from tqdm import tqdm

from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.trainers.trainer import Trainer
from moviad.scenarios.continual.strategies.replay.replay_memory import Memory
from moviad.utilities.evaluation.evaluator import Evaluator

class Replay(ContinualModel):

    def __init__(self, vad_model: VADModel, memory_size: int = 100, replay_ratio=0.5):
        super().__init__(vad_model)
        self.memory = Memory(memory_size=memory_size)
        self.replay_ratio = replay_ratio

    def start_task(self):
        pass

    def train_task(self, task_index: int, train_dataset, eval_dataset, train_args, metrics, device, logger):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=4
        ) if eval_dataset is not None else None

        train_args.init_train(self.vad_model)
        best_metrics = {metric.name: 0.0 for metric in metrics}

        # Calcola quanti campioni di replay inserire per batch
        n_replay_samples = int(train_args.batch_size * self.replay_ratio)

        for epoch in range(train_args.epochs):

            self.vad_model.train()

            print(f"EPOCH: {epoch}")

            avg_batch_loss = 0.0

            for batch in tqdm(train_dataloader):

                if task_index > 0:

                    # get replay samples from memory
                    memory_samples = self.memory.get_samples(min(n_replay_samples, batch.size(0)))
                    memory_samples = memory_samples.to(batch.device)

                    # replace a random subset of the current batch with replay samples
                    replace_idx = torch.randperm(batch.size(0))[:memory_samples.size(0)]
                    batch[replace_idx] = memory_samples

                avg_batch_loss += self.vad_model.train_step(batch, train_args)

            avg_batch_loss /= len(train_dataloader)

            if logger:
                logger.log({
                    f"Task_T{task_index}/train/epoch": epoch,
                    f"Task_T{task_index}/train/train_loss": avg_batch_loss,
                })

            print(f"Task_T{task_index}/train/epoch: {epoch}")
            print(f"Task_T{task_index}/train/train_loss: {avg_batch_loss}")

            if (epoch + 1) % train_args.evaluation_epoch_interval == 0:
                print("Evaluating model...")
                results = Evaluator.evaluate(self.vad_model, eval_dataloader, metrics, device)

                # TBD: save the model if needed

                # update best metrics
                best_metrics = Trainer.update_best_metrics(best_metrics, results)

                print("Training performances:")
                Trainer.print_metrics(results)

                if logger is not None:
                    logger.log({
                        f"Task_T{task_index}/train/{metric_name}": value for metric_name, value in results.items()
                    })

        for batch in train_dataloader:
            self.memory.add_samples(task_id=task_index, samples=batch)

    def end_task(self, task_index: int, train_dataset, train_args):
        pass
