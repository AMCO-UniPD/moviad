import math
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

# Adjust these imports based on your exact project paths
from moviad.datasets.vad_dataset import VADDataset
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.utilities.configurations import Split, LabelName
from moviad.datasets.dataset_arguments import DatasetArguments

IMG_EXTENSIONS = (".png", ".PNG")

CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)

class TestType(str, Enum):
    """Type of test set to use for MVTec AD 2."""
    PUBLIC = "public"
    PRIVATE = "private"
    PRIVATE_MIXED = "private_mixed"

class MVTecAD2Dataset(VADDataset):
    """MVTec AD 2 dataset class adapted for moviad."""

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split],
        test_type: TestType = TestType.PUBLIC
    ) -> None:
        super().__init__(
            arguments=dataset_arguments,
            category=category,
            split=split
        )

        self.test_type = test_type
        self.root_category = Path(self.dataset_arguments.dataset_path) / Path(self.category)
        self.samples: pd.DataFrame = None

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

        self.load_dataset()

    def is_loaded(self) -> bool:
        return self.samples is not None

    def load_dataset(self) -> None:
        if self.is_loaded():
            print("Dataset already loaded")
            return

        root = Path(self.root_category)
        samples_list = []

        # Ensure directory contains valid files before proceeding
        image_files = [f for f in root.glob("**/*") if f.suffix in IMG_EXTENSIONS]
        if not image_files:
            msg = f"Found 0 images in {root}. Make sure extensions are {IMG_EXTENSIONS}"
            raise RuntimeError(msg)

        # 1. Process Training Samples (Normal only)
        train_path = root / "train" / "good"
        if train_path.exists():
            train_samples = [
                (str(root), Split.TRAIN.value, "good", str(f), None, LabelName.NORMAL.value) 
                for f in train_path.glob(f"*[{''.join(IMG_EXTENSIONS)}]")
            ]
            samples_list.extend(train_samples)

        # 2. Process Validation Samples (Normal only)
        val_path = root / "validation" / "good"
        if val_path.exists():
            val_samples = [
                (str(root), Split.VAL.value, "good", str(f), None, LabelName.NORMAL.value) 
                for f in val_path.glob(f"*[{''.join(IMG_EXTENSIONS)}]")
            ]
            samples_list.extend(val_samples)

        # 3. Process Test Samples based on TestType
        if self.test_type == TestType.PUBLIC:
            test_path = root / "test_public"
            if test_path.exists():
                # Normal test samples
                test_normal_path = test_path / "good"
                test_normal_samples = [
                    (str(root), Split.TEST.value, "good", str(f), None, LabelName.NORMAL.value) 
                    for f in test_normal_path.glob(f"*[{''.join(IMG_EXTENSIONS)}]")
                ]
                samples_list.extend(test_normal_samples)

                # Abnormal test samples
                test_abnormal_path = test_path / "bad"
                if test_abnormal_path.exists():
                    for image_path in test_abnormal_path.glob(f"*[{''.join(IMG_EXTENSIONS)}]") :
                        # MVTec AD 2 adds '_mask' suffix to the mask filename
                        mask_name = image_path.stem + "_mask" + image_path.suffix
                        mask_path = root / "test_public" / "ground_truth" / "bad" / mask_name
                        
                        if not mask_path.exists():
                            msg = f"Missing mask for anomalous image: {image_path}"
                            raise RuntimeError(msg)
                        
                        samples_list.append(
                            (str(root), Split.TEST.value, "bad", str(image_path), str(mask_path), LabelName.ABNORMAL.value)
                        )

        elif self.test_type in [TestType.PRIVATE, TestType.PRIVATE_MIXED]:
            # Both private test sets don't have ground truth masks and are treated as 'unknown' (-1)
            test_dir_name = "test_private" if self.test_type == TestType.PRIVATE else "test_private_mixed"
            test_path = root / test_dir_name
            if test_path.exists():
                test_samples = [
                    (str(root), Split.TEST.value, "unknown", str(f), None, -1) 
                    for f in test_path.glob(f"*[{''.join(IMG_EXTENSIONS)}]")
                ]
                samples_list.extend(test_samples)

        # Create DataFrame
        samples = pd.DataFrame(
            samples_list, 
            columns=["path", "split", "label", "image_path", "mask_path", "label_index"]
        )

        # Filter by the requested split(s)
        if isinstance(self.split, list):
            split_values = [s.value for s in self.split]
            self.samples = samples[samples.split.isin(split_values)].reset_index(drop=True)
        else:
            self.samples = samples[samples.split == self.split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index of the element to be returned
        Returns:
            image (Tensor): tensor of shape (C,H,W) with values in [0,1]
            label (int): label of the image (if not train split)
            mask (Tensor): tensor of shape (1,H,W) with values in [0,1] (if not train split)
            path (str): path of the input image (if not train split)
        """
        if self.samples is None:
            self.load_dataset()

        sample_row = self.samples.iloc[index]
        image_path = sample_row.image_path

        # Open and transform the image
        image = self.transform_image(Image.open(image_path).convert("RGB"))

        if self.split == Split.TRAIN:
            return image
        
        # Validation / Test processing
        label = sample_row.label_index

        if label == LabelName.ABNORMAL.value and sample_row.mask_path:
            # Has a valid mask (Public Test Set)
            mask = Image.open(sample_row.mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            # Normal samples or Unknown test sets (Private / Mixed) return empty mask
            mask = torch.zeros(1, *self.dataset_arguments.gt_mask_size)

        return image, label, mask.int(), image_path

    def contaminate(self, source: VADDataset, ratio: float, seed: int = 42) -> int:
        """Contaminates the current dataset with abnormal samples from a source dataset."""
        if not isinstance(source, MVTecAD2Dataset):
            raise ValueError("Dataset should be of type MVTecAD2Dataset")
        if self.samples is None:
            raise ValueError("Destination dataset is not loaded")
        if source.samples is None:
            raise ValueError("Source dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(math.floor(len(self.samples) * ratio))
        contaminated_entries_indices = source.samples[source.samples["label_index"] == LabelName.ABNORMAL.value].index
        
        if len(contaminated_entries_indices) < contamination_set_size:
            raise DatasetTooSmallToContaminateException(
                f"Source dataset does not contain enough abnormal entries to contaminate the destination dataset. "
                f"Source dataset contains {len(contaminated_entries_indices)} abnormal entries, "
                f"while {contamination_set_size} are required."
            )

        contaminated_entries_indices = np.random.choice(
            contaminated_entries_indices, 
            contamination_set_size,
            replace=False
        )

        for index in contaminated_entries_indices:
            entry_metadata = source.samples.iloc[index]
            
            # Using hasattr checks to gracefully handle different loading strategies
            if hasattr(source, 'preload_imgs') and source.preload_imgs and hasattr(source, 'data'):
                entry = source.data[index]
                if hasattr(self, 'data'):
                    self.data.append(entry)
            else:
                entry = self.transform_image(
                    Image.open(source.samples.iloc[index].image_path).convert("RGB")
                )
                if hasattr(self, 'data'):
                    self.data.append(entry)
                
                if hasattr(source, 'data'):
                    source.data = [e for e in source.data if hash(e) != hash(entry)]

            self.samples = pd.concat([self.samples, pd.DataFrame([entry_metadata])], ignore_index=True)

        source.samples = source.samples.drop(contaminated_entries_indices).reset_index(drop=True)
        if hasattr(source, 'data'):
            source.data = [source.data[i] for i in range(len(source.data)) if i not in contaminated_entries_indices]
            
        return contamination_set_size

    def split_dataset(self, train_size, valid_size):
        """Implement if your pipeline dynamically splits datasets instead of using predefined folders."""
        pass

    def compute_contamination_ratio(self) -> float:
        """Calculate the ratio of abnormal vs total samples currently loaded."""
        if self.samples is None or len(self.samples) == 0:
            return 0.0
        abnormal_count = len(self.samples[self.samples["label_index"] == LabelName.ABNORMAL.value])
        return abnormal_count / len(self.samples)

    @staticmethod
    def get_categories() -> list:
        return list(CATEGORIES)