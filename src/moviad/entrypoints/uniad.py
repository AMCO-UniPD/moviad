"""
UniAD entrypoint for MoViAD.
Follows the same pattern as other MoViAD entrypoints (patchcore.py, rd4ad.py).
"""

import gc
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from moviad.entrypoints.common import load_datasets
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.uniad.uniad import UniAD, UniADTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.utilities.evaluation.metrics import RocAuc, MetricLvl
from moviad.datasets.builder import DatasetConfig, DatasetType


@dataclass
class UniADArgs:
    """Arguments for UniAD training."""
    dataset_config: DatasetConfig
    dataset_type: DatasetType
    category: str
    backbone: str = "efficientnet_b4"
    ad_layers: List[str] = field(default_factory=lambda: ["features.1", "features.2", "features.3", "features.5"])
    img_input_size: Tuple[int, int] = (224, 224)
    feature_size: List[int] = field(default_factory=lambda: [14, 14])
    batch_size: int = 4 # default lowered for testing
    epochs: int = 50 # default lowered for testing
    lr: float = 1e-4
    weight_decay: float = 1e-4
    neighbor_mask: Optional[dict] = field(default_factory=lambda: {
        "neighbor_size": [7, 7],
        "mask": [True, True, True]
    })
    feature_jitter: Optional[dict] = field(default_factory=lambda: {
        "scale": 20.0,
        "prob": 1.0
    })
    use_mfcn: bool = True
    mfcn_instrides: Optional[List[int]] = field(default_factory=lambda: [2, 4, 8, 16])
    device: torch.device = None
    save_path: Optional[str] = None


def train_uniad(args: UniADArgs, logger=None):
    """Train UniAD model following MoViAD's entrypoint pattern."""
    train_dataset, test_dataset = load_datasets(
        args.dataset_config,
        args.dataset_type,
        args.category,
        image_size=args.img_input_size
    )

    feature_extractor = CustomFeatureExtractor(
        args.backbone,
        args.ad_layers,
        args.device,
        True,
        False,
        None
    )

    model = UniAD(
        feature_extractor=feature_extractor,
        input_size=args.img_input_size,
        feature_size=args.feature_size,
        neighbor_mask=args.neighbor_mask,
        feature_jitter=args.feature_jitter,
        use_mfcn=args.use_mfcn,
        mfcn_instrides=args.mfcn_instrides,
    )
    model.to(args.device)

    train_args = UniADTrainArgs(
        batch_size=args.batch_size,
        epochs=args.epochs,
        evaluation_epoch_interval=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics = [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL)]

    trainer = Trainer(
        train_args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metrics=metrics,
        device=args.device,
        logger=logger,
        save_path=args.save_path,
    )

    trainer.train()

    del model
    torch.cuda.empty_cache()
    gc.collect()