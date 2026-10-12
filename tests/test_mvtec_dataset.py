"""Small filesystem fixtures for the MVTec dataset contract."""

from __future__ import annotations

import numpy as np
from PIL import Image

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec import MVTecDataset
from moviad.utilities.configurations import Split


def _write_image(path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def _mvtec_fixture(tmp_path):
    category = tmp_path / "bottle"
    _write_image(category / "train" / "good" / "000.png", 0)
    _write_image(category / "train" / "good" / "001.png", 32)
    _write_image(category / "test" / "good" / "000.png", 0)
    _write_image(category / "test" / "crack" / "001.png", 255)
    _write_image(category / "ground_truth" / "crack" / "001_mask.png", 255)
    return DatasetArguments(
        dataset_path=str(tmp_path), img_size=(4, 4), gt_mask_size=(2, 2)
    )


def test_string_test_split_loads_labels_and_masks(tmp_path):
    dataset = MVTecDataset(_mvtec_fixture(tmp_path), "bottle", "test")

    assert dataset.split is Split.TEST
    assert len(dataset) == 2
    samples = [dataset[index] for index in range(len(dataset))]
    normal = next(sample for sample in samples if sample[1] == 0)
    abnormal = next(sample for sample in samples if sample[1] == 1)
    assert normal[2].shape == (1, 2, 2)
    assert abnormal[2].shape == (1, 2, 2)
    assert normal[2].count_nonzero() == 0
    assert abnormal[2].count_nonzero() == 4


def test_contamination_moves_deterministic_abnormal_metadata(tmp_path):
    arguments = _mvtec_fixture(tmp_path)
    destination = MVTecDataset(arguments, "bottle", Split.TRAIN)
    source = MVTecDataset(arguments, "bottle", Split.TEST)

    moved = destination.contaminate(source, ratio=0.5, seed=7)

    assert moved == 1
    assert len(destination) == 3
    assert len(source) == 1
    assert destination.compute_contamination_ratio() == 1 / 3
