from abc import abstractmethod
from enum import Enum

from torch.utils.data.dataset import Dataset
from moviad.utilities.configurations import TaskType, Split

class IadDataset(Dataset):
    task : TaskType
    split: Split
    class_name: str
    dataset_path: str
    contamination_ratio: float

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def set_category(self, category: str):
        self.class_name = category

    @abstractmethod
    def compute_contamination_ratio(self) -> float:
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def contaminate(self, source: 'IadDataset', ratio: float, seed: int = 42) -> int:
        pass

    @abstractmethod
    def contains(self, entry) -> bool:
        pass
