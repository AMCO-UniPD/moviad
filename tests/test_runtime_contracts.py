"""Deterministic regressions for shared runtime contracts."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
from moviad.trainers.trainer import Trainer
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.evaluator import min_max_norm
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc


class _TensorDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index + 1)])


class _TinyModel(VADModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs) -> float:
        return float((batch * self.weight).mean().detach())

    def train_chunk(self, train_dataloader, training_args: TrainingArgs) -> float:
        return self.train_epoch(0, train_dataloader, training_args)


class _ContinualModel(ContinualModel):
    def start_task(self, task_index, train_dataset, train_args=None):
        return None

    def train_task(
        self,
        task_index,
        train_dataset,
        eval_dataset,
        metrics,
        device,
        logger=None,
        train_args=None,
    ):
        return None

    def end_task(self, task_index, train_dataset, train_args=None):
        return None


class _AnomalyDataset(Dataset):
    def __init__(self, arguments=None, category="one", split=Split.TRAIN):
        self.category = category
        self.values = torch.tensor([0.0, 1.0, 0.0, 1.0])

    @staticmethod
    def get_categories() -> list[str]:
        return ["one", "two"]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        label = self.values[index]
        image = torch.full((1, 2, 2), label)
        mask = torch.full((1, 2, 2), label)
        return image, label, mask, f"{self.category}-{index}.png"


class _AnomalyModel(VADModel):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, images: torch.Tensor):
        maps = images * self.scale
        return maps, maps.flatten(1).mean(1)

    def train_step(self, batch, training_args: TrainingArgs) -> float:
        images, labels, _, _ = batch
        self.optimizer.zero_grad()
        _, scores = self(images)
        loss = torch.nn.functional.mse_loss(scores, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_chunk(self, train_dataloader, training_args: TrainingArgs) -> float:
        return self.train_epoch(0, train_dataloader, training_args)


def test_package_and_model_modules_are_lazy():
    script = """
import sys
import moviad
assert 'torch' not in sys.modules
import moviad.models
assert 'torch' not in sys.modules
assert moviad.models.VADModel.__name__ == 'VADModel'
assert 'torch' in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_constant_min_max_normalization_is_finite():
    result = min_max_norm(np.full((2, 3), 4.0))

    assert np.array_equal(result, np.zeros((2, 3)))
    assert np.isfinite(result).all()


def test_trainer_can_save_without_an_evaluation_dataset(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    model = _TinyModel()
    args = TrainingArgs(batch_size=1, epochs=1, num_workers=0)
    trainer = Trainer(
        train_args=args,
        model=model,
        train_dataset=_TensorDataset(),
        eval_dataset=None,
        metrics=[],
        device=torch.device("cpu"),
        save_path=str(checkpoint),
    )

    trainer.train()

    restored = _TinyModel()
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    assert torch.equal(restored.weight, model.weight)


def test_continual_model_mode_methods_follow_torch_conventions():
    model = _ContinualModel(_TinyModel())

    assert model.train(False) is model
    assert model.training is False
    assert model.vad_model.training is False
    assert model.train() is model
    assert model.training is True
    assert model.eval() is model
    assert model.training is False


def test_synthetic_continual_training_lifecycle():
    metric = RocAuc(MetricLvl.IMAGE)
    args = TrainingArgs(batch_size=2, epochs=1, num_workers=0)
    stream = ContinualDataset(DatasetArguments("/unused"), _AnomalyDataset)
    trainer = ContinualTrainer(
        continual_dataset=stream,
        continual_model=FineTuning(_AnomalyModel()),
        device=torch.device("cpu"),
        metrics=[metric],
        training_args=args,
    )

    matrices = trainer.train()

    assert matrices[metric.name].shape == (2, 2)
    observed = matrices[metric.name][~np.isnan(matrices[metric.name])]
    assert np.array_equal(observed, np.ones_like(observed))
