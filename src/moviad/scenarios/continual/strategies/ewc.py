from functools import partial

import torch
from tqdm import tqdm

from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric
from moviad.trainers.trainer import Trainer

class EWC(ContinualModel):
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017)

    The penalty  lambda/2 * sum_i F_i * (theta_i - theta*_i)^2  is applied by adding its gradient
    lambda * F_i * (theta_i - theta*_i) to the parameters gradients through backward hooks, so that
    the train_step of the wrapped VADModel does not need to be modified.

    The importances (diagonal empirical Fisher) are computed at the end of every task with
    VADModel.compute_loss, only on the trainable parameters (the student), since the
    pretrained encoder (teacher) is kept frozen.

    Args:
        model (VADModel): model to be trained
        ewc_lambda (float): weight of the EWC penalty
        mode (str): "separate" keeps one penalty for each past task, "online" keeps a single
            penalty with importances accumulated as F = decay_factor * F_old + F_new
        decay_factor (float): decay of the old importances, used only in "online" mode
    """

    def __init__(self, model: VADModel, ewc_lambda: float = 1.0, mode: str = "separate", decay_factor: float = 1.0):
        super().__init__(model)

        if mode not in ["separate", "online"]:
            raise ValueError(f"Mode {mode} not supported. Available modes: [separate, online]")

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        # one dict {param_name: tensor} for each past task (a single one in online mode)
        self.importances: list[dict[str, torch.Tensor]] = []
        self.saved_params: list[dict[str, torch.Tensor]] = []

        self.hook_handles = []

    def get_ewc_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        """
        Return the parameters regularized by EWC: the trainable ones, the pretrained encoder is frozen
        """
        return [(name, param) for name, param in self.vad_model.named_parameters() if param.requires_grad]

    def penalty_grad(self, name: str, param: torch.Tensor) -> torch.Tensor:
        theta = param.detach()
        return sum(
            importances[name] * (theta - saved_params[name])
            for importances, saved_params in zip(self.importances, self.saved_params)
        )

    def penalty_hook(self, name: str, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return grad + self.ewc_lambda * self.penalty_grad(name, param)

    def compute_penalty(self) -> float:
        penalty = 0.0
        with torch.no_grad():
            for name, param in self.get_ewc_parameters():
                for importances, saved_params in zip(self.importances, self.saved_params):
                    penalty += (importances[name] * (param - saved_params[name]) ** 2).sum().item()
        return 0.5 * self.ewc_lambda * penalty

    def register_hooks(self):
        for name, param in self.get_ewc_parameters():
            self.hook_handles.append(param.register_hook(partial(self.penalty_hook, name, param)))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def compute_importances(self, train_dataset: VADDataset, train_args: TrainingArgs) -> dict[str, torch.Tensor]:
        """
        Compute the diagonal empirical Fisher information as the mean of the squared mini-batch gradients
        """
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=4
        )

        # loss function needed by compute_loss
        train_args.init_train(self.vad_model)

        ewc_params = self.get_ewc_parameters()
        importances = {name: torch.zeros_like(param) for name, param in ewc_params}

        self.vad_model.train()

        for batch in tqdm(train_dataloader, desc="Computing EWC importances"):
            for _, param in ewc_params:
                param.grad = None

            loss = self.vad_model.compute_loss(batch, train_args)
            loss.backward()

            for name, param in ewc_params:
                if param.grad is not None:
                    importances[name] += param.grad.detach() ** 2

        for name in importances.keys():
            importances[name] /= len(train_dataloader)

        for _, param in ewc_params:
            param.grad = None

        return importances

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs):
        if task_index > 0:
            self.register_hooks()

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
        )

        trainer.train()

        if task_index > 0:
            penalty = self.compute_penalty()
            print(f"Task_T{task_index}/train/ewc_penalty: {penalty}")
            if logger:
                logger.log({f"Task_T{task_index}/train/ewc_penalty": penalty})

    def end_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs):
        # the penalty must not contribute to the importances of the current task
        self.remove_hooks()

        importances = self.compute_importances(train_dataset, train_args)
        saved_params = {name: param.detach().clone() for name, param in self.get_ewc_parameters()}

        if self.mode == "separate" or task_index == 0:
            self.importances.append(importances)
            self.saved_params.append(saved_params)
        else:
            self.importances = [{
                name: self.decay_factor * self.importances[0][name] + importances[name]
                for name in importances.keys()
            }]
            self.saved_params = [saved_params]
