import copy
from typing import Callable

import torch

from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.models.rd4ad.loss_functions import rd4ad_cosine_loss
from moviad.models.dinomaly.dinomaly import Dinomaly
from moviad.models.stfpm.stfpm import STFPM
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric
from moviad.trainers.trainer import Trainer


# select the student features from the outputs of the forward in training mode
STUDENT_FEATURES = {
    RD4AD: lambda outputs: outputs[2],     # (encoder, bottleneck, decoder) features
    Dinomaly: lambda outputs: outputs[1],  # (encoder, decoder) features
    STFPM: lambda outputs: outputs[1],     # (teacher, student) features
}


class _AddLossGradient(torch.autograd.Function):
    """
    Identity on x that, during the backward, propagates a unit gradient to the scalar loss,
    so that the gradient of loss is added to the parameters gradients exactly once
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, loss: torch.Tensor):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, torch.ones((), device=grad_output.device, dtype=grad_output.dtype)


class LwF(ContinualModel):
    """
    Learning without Forgetting (Li & Hoiem, 2017) adapted to teacher-student VAD models

    At the end of every task the model is frozen and kept as old model. While training on a new task,
    the features produced by the student on the new data are distilled towards the ones produced by
    the old student:  loss = task_loss + lambda * mean_l D(old_student_l, student_l)

    The distillation term is computed wrapping the forward of the VADModel and its gradient is
    injected in the autograd graph, so that the train_step of the wrapped VADModel does not need to be
    modified (custom gradient manipulations of the task loss, e.g. Dinomaly hard mining, do not affect it).

    The frozen parameters (not optimized by train_args.optimizer, e.g. the teacher encoder) are shared
    between the current and the old model.

    Args:
        model (VADModel): model to be trained
        lwf_lambda (float): weight of the distillation loss
        distillation_loss (Callable): loss between the old and the new student features of a layer,
            called as distillation_loss(old_features, new_features). Default: cosine distance (RD4AD loss)
        student_features_fn (Callable): function that returns the list of student features from the
            outputs of the forward in training mode. The returned list must be the one contained in the
            outputs. Default: defined for RD4AD, Dinomaly and STFPM
    """

    def __init__(self,
                 model: VADModel,
                 lwf_lambda: float = 1.0,
                 distillation_loss: Callable | None = None,
                 student_features_fn: Callable | None = None):
        super().__init__(model)

        if student_features_fn is None:
            if type(model) not in STUDENT_FEATURES:
                raise ValueError(f"Model {model.__class__.__name__} not supported. Available models: "
                                 f"[{', '.join(m.__name__ for m in STUDENT_FEATURES)}], "
                                 f"otherwise provide student_features_fn")
            student_features_fn = STUDENT_FEATURES[type(model)]

        self.lwf_lambda = lwf_lambda
        self.distillation_loss = distillation_loss if distillation_loss is not None else rd4ad_cosine_loss
        self.student_features_fn = student_features_fn

        self.old_model: VADModel | None = None

        self.avg_distillation_loss = 0.0
        self.n_distillation_steps = 0

    def compute_distillation(self, new_features: list[torch.Tensor], old_features: list[torch.Tensor]) -> torch.Tensor:
        loss = 0
        for new, old in zip(new_features, old_features):
            loss += self.distillation_loss(old, new)
        return loss / len(new_features)

    def forward_with_distillation(self, original_forward: Callable, batch: torch.Tensor):
        outputs = original_forward(batch)

        # distill only during the training steps
        if not (self.vad_model.training and torch.is_grad_enabled()):
            return outputs

        with torch.no_grad():
            old_outputs = self.old_model(batch)

        new_features = self.student_features_fn(outputs)
        old_features = self.student_features_fn(old_outputs)

        loss = self.compute_distillation(new_features, old_features)

        self.avg_distillation_loss += loss.item()
        self.n_distillation_steps += 1

        # the task loss receives new tensors, so its gradient hooks do not modify the distillation gradient
        new_features[0] = _AddLossGradient.apply(new_features[0], self.lwf_lambda * loss)
        for i in range(1, len(new_features)):
            new_features[i] = new_features[i].view_as(new_features[i])

        return outputs

    def register_distillation(self):
        original_forward = self.vad_model.forward
        self.vad_model.forward = lambda batch: self.forward_with_distillation(original_forward, batch)

    def remove_distillation(self):
        # remove the instance attribute, restoring the class forward
        self.vad_model.__dict__.pop("forward", None)

    def freeze_old_model(self, train_args: TrainingArgs) -> VADModel:
        """
        Copy the current model sharing the parameters not optimized by the optimizer
        """
        optimized_params = {id(p) for group in train_args.optimizer.param_groups for p in group["params"]}
        memo = {id(p): p for p in self.vad_model.parameters() if id(p) not in optimized_params}
        shared_params = set(memo.keys())

        old_model = copy.deepcopy(self.vad_model, memo)

        # deterministic outputs (no dropout, BatchNorm running stats) in the training format
        old_model.eval()
        old_model.training = True

        # the shared parameters keep their state, they belong also to the current model
        for param in old_model.parameters():
            if id(param) not in shared_params:
                param.requires_grad = False

        return old_model

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs):
        if task_index > 0:
            self.register_distillation()

        self.avg_distillation_loss = 0.0
        self.n_distillation_steps = 0

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
            avg_loss = self.avg_distillation_loss / max(self.n_distillation_steps, 1)
            print(f"Task_T{task_index}/train/distillation_loss: {avg_loss}")
            if logger:
                logger.log({f"Task_T{task_index}/train/distillation_loss": avg_loss})

    def end_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs):
        # the wrapped forward must not be copied in the old model
        self.remove_distillation()

        self.old_model = None
        self.old_model = self.freeze_old_model(train_args)
