from abc import abstractmethod
from enum import Enum
from dataclasses import dataclass
from collections.abc import Sequence

from torch.utils.data.dataset import Dataset
from torchvision import transforms

from moviad.utilities.configurations import TaskType, Split
from moviad.datasets.dataset_arguments import DatasetArguments
from torchvision.transforms.functional import InterpolationMode

class VADDataset(Dataset):
    """
    Args:
        split (Split): The split of the dataset (train, val, test).
        dataset_path (str): The path to the dataset.
        contamination_ratio (float): The ratio of contamination in the dataset.
    """

    #TODO: to be defined the splitting strategy for all the VAD datasets
    #TODO: add the contamination part

    def __init__(
        self,
        arguments: DatasetArguments,
        category: str,
        split: Split | str | Sequence[Split | str],
    ):
        self.dataset_arguments = arguments
        self.category = category
        if isinstance(split, (Split, str)):
            self.split = Split(split)
        else:
            self.split = [Split(item) for item in split]

        if self.dataset_arguments.image_transform_list:
            self.transform_image = transforms.Compose(self.dataset_arguments.image_transform_list)
        else:
            self.transform_image = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.dataset_arguments.img_size, antialias=True),
                ]
            )

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    self.dataset_arguments.gt_mask_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod 
    def split_dataset(self, train_size, valid_size): ...

    @staticmethod
    def get_categories() -> list: ...

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def __getitem__(self, index): ...

    @abstractmethod
    def contaminate(self, ratio: float, seed: int = 42) -> int: ...

    @abstractmethod
    def compute_contamination_ratio(self) -> float: ...
