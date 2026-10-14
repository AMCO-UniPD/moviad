import torch
from dataclasses import dataclass

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import Split

@dataclass
class Args:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    contamination_ratio: float = 0.0
    seed: int = 4
    dataset_config: DatasetArguments | None = None
    dataset_type: type[VADDataset] | None = None
    batch_size: int = 4
