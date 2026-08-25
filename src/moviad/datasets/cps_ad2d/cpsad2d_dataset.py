from pathlib import Path

import pandas as pd
from PIL import Image
import torch

from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import Split, LabelName
from moviad.datasets.dataset_arguments import DatasetArguments

IMG_EXTENSIONS = (".png", ".PNG", ".jpg", ".JPG")

# Not real categories, just for parsing the dataset structure
CATEGORIES = (
    "cps_1", 
    "cps_2",   
    "cps_3",
    "cps_4",
    "cps_5",
    "cps_6",
)

IMG_SIZE = (3, 270, 480)

PATCH_SIZE = 224

"""
Create CPS-AD2D samples by parsing the CPS-AD2D data file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/images/train/image_filename.jpg
        path/to/dataset/category/annotation/train/mask_filename.png

"""

class CPSAD2DDataset(VADDataset):

    """CPS-AD2D dataset class."""

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        split: Split | list[Split],
        category: str | None = None
    ) -> None:

        super().__init__(
            dataset_arguments,
            category,
            split
        )

        self.dataset_root = Path(self.dataset_arguments.dataset_path)
        self.samples: pd.DataFrame = None
        self.category = "ceramic package substrate"
        self.load_dataset()
    
    def is_loaded(self) -> bool:
        return self.samples is not None
    
    def load_dataset(self):

        if self.is_loaded():
            print("Dataset is already loaded.")
            return
        
        all_samples = []
        
        for category_path in self.dataset_root.iterdir():

            if not category_path.is_dir(): continue
            
            category_name = category_path.name
            images_path = category_path / "images" / "train"
            annotations_path = category_path / "annotations" / "train"

            if not images_path.exists(): continue

            for image_path in images_path.glob("*"):
                if image_path.suffix in IMG_EXTENSIONS:
                    
                    # Special if for cps_6 folder 
                    if category_name == "cps_6":
                        mask_path = annotations_path.parent / "defect" / (image_path.stem + ".png")
                    else:
                        mask_path = annotations_path / (image_path.stem + ".png")

                    if not mask_path.exists():
                        continue 

                    # Open mask and split into two patches (left and right)
                    with Image.open(mask_path) as mask_raw:
                        
                        mask_gray = mask_raw.convert("L")
                        
                        w, h = mask_gray.size # 480, 270
                        mid = w // 2         # 240
                        
                        # Define the boxes for the left and right patches
                        parts = [
                            (0, 0, mid, h),      # Sx
                            (mid, 0, w, h)       # Dx
                        ]

                        for box in parts:

                            # Cut mask
                            patch_mask = mask_gray.crop(box)
                                                        
                            # Determine label based on mask content
                            if patch_mask.getextrema()[1] > 0:
                                current_label = LabelName.ABNORMAL
                            else:
                                current_label = LabelName.NORMAL

                            all_samples.append({
                                "image_path": str(image_path),
                                "mask_path": str(mask_path),
                                "label": current_label,
                                "category": category_name,
                                "split": "Undefined",  # Placeholder, will be set in _apply_custom_split
                                "patch_box": box      
                            })

        df_samples = pd.DataFrame(all_samples)

        if df_samples.empty:
            raise RuntimeError(f"No image in : {self.dataset_root}")

        # Custom split
        self.samples = self._apply_custom_split(df_samples, train_ratio=0.75)


    def _apply_custom_split(self, df: pd.DataFrame, train_ratio: float):
        
        seed = 42 
        
        # Split normal and anomalous samples
        is_normal = df['label'] == LabelName.NORMAL
        df_normal_all = df[is_normal]
        df_anomaly_all = df[~is_normal]

        num_train_normal = int(len(df_normal_all) * train_ratio)
        
        
        df_normal_shuffled = df_normal_all.sample(frac=1, random_state=seed)
        train_indices = df_normal_shuffled.index[:num_train_normal]
        
        # Rest normal samples for test
        remaining_normal_indices = df_normal_shuffled.index[num_train_normal:]
        num_test_normal = len(remaining_normal_indices)
        
        # Anomalies for test
        if len(df_anomaly_all) >= num_test_normal:
            test_anomaly_indices = df_anomaly_all.sample(n=num_test_normal, random_state=seed).index
        else:
            remaining_normal_indices = remaining_normal_indices[:len(df_anomaly_all)] # To avoid imbalance in test set
            test_anomaly_indices = df_anomaly_all.sample(frac=1, random_state=seed).index
        
        test_indices = list(remaining_normal_indices) + list(test_anomaly_indices)

        # Set split labels
        df['split'] = 'excluded' 

        df.loc[train_indices, 'split'] = "train"
        df.loc[test_indices, 'split'] = "test"

        # Return only samples for the target split
        target_split = self.split.value if hasattr(self.split, 'value') else self.split
        return df[df['split'] == target_split].reset_index(drop=True)
    
    def __getitem__(self, index: int):

            """
            Args:
                index (int): indice dell'elemento da restituire
            Returns:
                TRAIN: image, 
                TEST: image, label, mask, path
            """

            if self.samples is None:
                self.load_dataset()

            sample = self.samples.iloc[index]
            
            # Load image and apply patch cropping
            full_img = Image.open(sample.image_path).convert("RGB")
            patch_img = full_img.crop(sample.patch_box)
            image = self.transform_image(patch_img)

            label = sample.label
            path = str(sample.image_path)

            # Mask handling
            if label == LabelName.NORMAL or sample.mask_path is None:

                mask = torch.zeros(1, image.shape[1], image.shape[2])
            else:

                full_mask = Image.open(sample.mask_path).convert("L")
                patch_mask = full_mask.crop(sample.patch_box)
                mask = self.transform_mask(patch_mask)

            if self.split == Split.TRAIN:
                return image
            else:
                return image, label, mask.int(), path
        
    def __len__(self) -> int:

        if self.samples is None:
            return 0
            
        return len(self.samples)
            
            
