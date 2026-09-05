"""
This entrypoint reproduces the UniAD model unified training approach:
a single model is trained on all MVTec-AD categories together, then
evaluated separately per category. To speed up training, backbone
features are extracted and cached before the training loop starts,
so the frozen backbone runs only once instead of per epoch.
"""

import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.uniad.uniad import UniAD, UniADTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.metrics import Metric, MetricLvl, RocAuc


def _min_max_norm(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


@dataclass
class UniADUnifiedArgs:
    """Config for the paper's unified (multi-class) MVTec-AD experiment."""

    dataset_path: str
    categories: Optional[List[str]] = None  #all 15 MVTec-AD categories
    backbone: str = "efficientnet_b4"
    ad_layers: List[str] = field(default_factory=lambda: ["features.1", "features.2", "features.3", "features.5"])
    img_input_size: Tuple[int, int] = (224, 224)
    feature_size: List[int] = field(default_factory=lambda: [14, 14])
    use_mfcn: bool = True
    mfcn_instrides: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    neighbor_mask: dict = field(default_factory=lambda: {"neighbor_size": [7, 7], "mask": [True, True, True]})
    feature_jitter: dict = field(default_factory=lambda: {"scale": 20.0, "prob": 1.0})
    avgpool_size: int = 16
    batch_size: int = 4
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    step_size: int = 800
    gamma: float = 0.1
    eval_every: int = 10
    cache_batch_size: int = 64
    device: torch.device = None
    save_path: Optional[str] = None


def _cache_train_features(model: UniAD, dataset, device: torch.device, batch_size: int) -> TensorDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features = []
    model.eval()
    for images in tqdm(loader, desc="Caching train features"):
        with torch.no_grad():
            features.append(model._extract_features(images.to(device)).cpu())
    return TensorDataset(torch.cat(features, dim=0))


def _cache_test_features(model: UniAD, dataset, device: torch.device, batch_size: int) -> TensorDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features, labels, masks = [], [], []
    model.eval()
    for images, label, mask, _path in tqdm(loader, desc="Caching test features"):
        with torch.no_grad():
            features.append(model._extract_features(images.to(device)).cpu())
        labels.append(label)
        masks.append(mask)
    return TensorDataset(torch.cat(features, dim=0), torch.cat(labels, dim=0), torch.cat(masks, dim=0))


def _evaluate_from_cache(model: UniAD, dataloader: DataLoader, metrics: List[Metric], device: torch.device) -> dict:
    model.eval()
    gt_mask, gt_label, pred_map, pred_score = [], [], [], []
    for feature_align, label, mask in dataloader:
        with torch.no_grad():
            anomaly_maps, anomaly_scores = model.forward_from_features(feature_align.to(device))
        gt_mask.append(mask.cpu().numpy().astype(int))
        gt_label.append(label.cpu().numpy())
        pred_map.append(anomaly_maps.cpu().numpy())
        pred_score.append(anomaly_scores.cpu().numpy())

    gt_mask = np.concatenate(gt_mask)
    gt_label = np.concatenate(gt_label)
    pred_map = _min_max_norm(np.concatenate(pred_map))
    pred_score = np.concatenate(pred_score)

    report = {}
    for metric in metrics:
        gt, pred = (gt_label, pred_score) if metric.level == MetricLvl.IMAGE else (gt_mask, pred_map)
        report[metric.name] = metric.compute(gt, pred)
    return report


def train_uniad_unified(args: UniADUnifiedArgs, logger=None) -> Tuple[Dict[str, dict], dict]:
    """Train one UniAD model jointly on all categories, evaluate per category and average."""
    categories = args.categories or MVTecDataset.get_categories()
    dataset_args = DatasetArguments(
        dataset_path=args.dataset_path,
        img_size=args.img_input_size,
        gt_mask_size=args.img_input_size,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(args.img_input_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    train_datasets = [MVTecDataset(dataset_args, category=c, split=Split.TRAIN) for c in categories]
    joint_train_dataset = ConcatDataset(train_datasets)

    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device, True, False, None)
    model = UniAD(
        feature_extractor=feature_extractor,
        input_size=args.img_input_size,
        feature_size=args.feature_size,
        neighbor_mask=args.neighbor_mask,
        feature_jitter=args.feature_jitter,
        use_mfcn=args.use_mfcn,
        mfcn_instrides=args.mfcn_instrides,
        avgpool_size=args.avgpool_size,
    ).to(args.device)

    cached_train = _cache_train_features(model, joint_train_dataset, args.device, args.cache_batch_size)
    cached_test = {
        c: _cache_test_features(model, MVTecDataset(dataset_args, category=c, split=Split.TEST), args.device, args.cache_batch_size)
        for c in categories
    }

    train_loader = DataLoader(cached_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loaders = {c: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0) for c, ds in cached_test.items()}

    train_args = UniADTrainArgs(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
    )
    train_args.init_train(model)

    if logger is not None:
        logger.config.update(train_args.__to_dict__())

    metrics = [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL)]

    def evaluate_all_categories(epoch: int):
        per_category = {c: _evaluate_from_cache(model, loader, metrics, args.device) for c, loader in test_loaders.items()}
        mean_metrics = {m.name: float(np.mean([per_category[c][m.name] for c in categories])) for m in metrics}
        if logger is not None:
            log_dict = {f"{c}/{k}": v for c, r in per_category.items() for k, v in r.items()}
            log_dict.update({f"mean/{k}": v for k, v in mean_metrics.items()})
            log_dict["epoch"] = epoch
            logger.log(log_dict)
        return per_category, mean_metrics

    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        for (feature_align,) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            avg_loss += model.train_step_from_features(feature_align.to(args.device), train_args)
        avg_loss /= len(train_loader)

        if train_args.scheduler is not None:
            train_args.scheduler.step()

        if logger is not None:
            logger.log({"epoch": epoch, "train_loss": avg_loss})

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            _, mean_metrics = evaluate_all_categories(epoch)
            print(f"[epoch {epoch}] loss={avg_loss:.5f} mean={mean_metrics}")

    per_category, mean_metrics = evaluate_all_categories(args.epochs - 1)

    print("\nFinal per-category results:")
    for c in categories:
        print(f"  {c}: {per_category[c]}")
    print(f"Mean: {mean_metrics}")

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return per_category, mean_metrics