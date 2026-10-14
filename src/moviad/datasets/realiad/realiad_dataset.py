import math
import numpy as np
import torch
from typing import List, Optional
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torch.utils.data import Dataset
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.datasets.realiad.realiad_data import RealIadData
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum, RealIadAnomalyClass
from moviad.utilities.configurations import TaskType, Split, LabelName


class RealIadDataset(Dataset):
    def __init__(self, class_name: str, img_root_dir: str, json_root_path: str, task: TaskType, split: Split,
                 gt_mask_size: Optional[tuple] = None,
                 transform=None,
                 image_size=(224, 224)) -> None:
        super().__init__()
        if img_root_dir is None:
            raise ValueError("img_dir should not be None")
        if not os.path.exists(img_root_dir):
            raise ValueError(f"img_dir '{img_root_dir}' does not exist")
        if not os.path.isdir(img_root_dir):
            raise ValueError(f"img_dir '{img_root_dir}' is not a directory")
        self.json_root_path = json_root_path
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.category = class_name
        self.data: RealIadData = None
        self.task = task
        self.split = split
        self.gt_mask_size = gt_mask_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.PILToTensor(),
                transforms.Resize(
                    image_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ConvertImageDtype(torch.float32),
            ])

    def compute_contamination_ratio(self) -> float:
        if self.data is None:
            raise ValueError("Dataset is not loaded")

        return self.data.compute_contamination_ratio()

    def contaminate(self, source: 'RealIadDataset', ratio: float, seed: int = 42) -> int:
        if type(source) != RealIadDataset:
            raise ValueError("Dataset should be of type RealIadDataset")
        if self.data is None or self.data.data is None:
            raise ValueError("Destination dataset is not loaded")
        if source.data is None or source.data.data is None:
            raise ValueError("Source dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(math.floor(len(self.data) * ratio))
        contaminated_data_entries = [entry for entry in source.data.data if
                                     entry.anomaly_class != RealIadAnomalyClass.OK]
        contaminated_image_entries = [image for image in source.data.data if
                                      image.anomaly_class != RealIadAnomalyClass.OK]
        if len(contaminated_data_entries) < contamination_set_size:
            raise DatasetTooSmallToContaminateException(
                f"Source dataset does not have enough contaminated entries to contaminate the dataset. "
                f"Found {len(contaminated_data_entries)} entries, but needed {contamination_set_size} entries")

        contaminated_data_entries = np.random.choice(contaminated_data_entries, contamination_set_size,
                                                     replace=False).tolist()
        contaminated_image_entries = np.random.choice(contaminated_image_entries, contamination_set_size,
                                                      replace=False).tolist()
        self.data.data.extend(contaminated_data_entries)
        self.data.images.extend(contaminated_image_entries)
        source.data.data = [entry for entry in source.data.data if entry not in contaminated_data_entries]
        source.data.data = [image for image in source.data.data if image not in contaminated_image_entries]
        return contamination_set_size

    def partition(self, dataset: Dataset, ratio: float) -> ('RealIadDataset', 'RealIadDataset'):
        if not isinstance(dataset, RealIadDataset):
            raise ValueError("Dataset should be of type RealIadDataset")
        split_1, split_2 = self.data.partition(ratio)
        dataset_1 = self
        dataset_2 = self
        dataset_1.data = split_1
        dataset_2.data = split_2
        return dataset_1, dataset_2

    def load_dataset(self) -> None:
        self.data = RealIadData.from_json(self.json_root_path, self.category, self.split)
        if self.data is None:
            raise ValueError("Dataset is None")

        self.__index_images_and_labels__()

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, item):
        image_data, image_entry = self.data.__getitem__(item)

        if self.split == Split.TRAIN:
            if self.transform:
                image_entry = self.transform(image_entry.image)
            return image_entry

        if self.split == Split.TEST:
            image = self.transform(image_entry.image)
            label = LabelName.NORMAL.value if image_data.anomaly_class == RealIadAnomalyClass.OK else LabelName.ABNORMAL.value
            path = image_data.image_path
            if image_entry.mask is not None:
                mask = image_entry.mask
                mask = self.transform(mask)
            else:
                mask = torch.zeros(1, *self.gt_mask_size, dtype=torch.float32)

            return image, label, mask, path

    def __index_images_and_labels__(self) -> None:
        self.data.load_images(self.img_root_dir)
