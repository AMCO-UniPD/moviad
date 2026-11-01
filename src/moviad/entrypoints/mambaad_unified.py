"""
MambaAD "unified case" reproduction entrypoint for MoViAD.
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
from moviad.models.mambaad.mambaad import MambaAD, MambaADTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.metrics import RocAuc, AvgPrec, F1, ProAuc, MetricLvl


def _min_max_norm(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


@dataclass
class MambaADUnifiedArgs:
    dataset_path: str
    categories: Optional[List[str]] = None
    img_input_size: Tuple[int, int] = (256, 256)
    base_channels: int = 64
    decoder_depths: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    d_state: int = 16
    num_direction: int = 8
    use_checkpoint: bool = True
    chunk_size: int = 256
    batch_size: int = 16
    epochs: int = 50
    lr: float = 5e-3
    weight_decay: float = 1e-4
    loss_weight: float = 5.0
    use_amp: bool = True
    eval_every: int = 10
    cache_batch_size: int = 16
    device: torch.device = None
    save_path: Optional[str] = None


def _cache_train_features(model: MambaAD, dataset, device: torch.device, batch_size: int) -> TensorDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    per_scale_features = None
    model.eval()
    for images in tqdm(loader, desc="Caching train features"):
        with torch.no_grad():
            feats = model._extract_features(images.to(device))
        if per_scale_features is None:
            per_scale_features = [[] for _ in feats]
        for i, f in enumerate(feats):
            per_scale_features[i].append(f.cpu())
    return TensorDataset(*[torch.cat(f, dim=0) for f in per_scale_features])


def _cache_test_features(model: MambaAD, dataset, device: torch.device, batch_size: int) -> TensorDataset:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    per_scale_features, labels, masks = None, [], []
    model.eval()
    for images, label, mask, _path in tqdm(loader, desc="Caching test features"):
        with torch.no_grad():
            feats = model._extract_features(images.to(device))
        if per_scale_features is None:
            per_scale_features = [[] for _ in feats]
        for i, f in enumerate(feats):
            per_scale_features[i].append(f.cpu())
        labels.append(label)
        masks.append(mask)
    tensors = [torch.cat(f, dim=0) for f in per_scale_features]
    return TensorDataset(*tensors, torch.cat(labels, dim=0), torch.cat(masks, dim=0))


def _evaluate_from_cache(model: MambaAD, dataloader: DataLoader, metrics, device: torch.device) -> dict:
    model.eval()
    gt_mask, gt_label, pred_map, pred_score = [], [], [], []
    for batch in dataloader:
        *feats, label, mask = batch
        feats = [f.to(device) for f in feats]
        with torch.no_grad():
            anomaly_maps, anomaly_scores = model.forward_from_features(feats)
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


def train_mambaad_unified(args: MambaADUnifiedArgs, logger=None) -> Tuple[Dict[str, dict], dict]:
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

    model = MambaAD(
        input_size=args.img_input_size,
        base_channels=args.base_channels,
        decoder_depths=args.decoder_depths,
        d_state=args.d_state,
        num_direction=args.num_direction,
        use_checkpoint=args.use_checkpoint,
        chunk_size=args.chunk_size,
    )
    model.to(args.device)

    cached_train = _cache_train_features(model, joint_train_dataset, args.device, args.cache_batch_size)
    cached_test = {
        c: _cache_test_features(model, MVTecDataset(dataset_args, category=c, split=Split.TEST), args.device, args.cache_batch_size)
        for c in categories
    }

    train_loader = DataLoader(cached_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loaders = {c: DataLoader(ds, batch_size=args.batch_size, shuffle=False) for c, ds in cached_test.items()}

    train_args = MambaADTrainArgs(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_weight=args.loss_weight,
        use_amp=args.use_amp,
    )
    train_args.init_train(model)

    if logger is not None:
        logger.config.update(train_args.__to_dict__())

    metrics = [
        RocAuc(MetricLvl.IMAGE), AvgPrec(MetricLvl.IMAGE), F1(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL), AvgPrec(MetricLvl.PIXEL), F1(MetricLvl.PIXEL), ProAuc(MetricLvl.PIXEL),
    ]

    def evaluate_all_categories(epoch: int):
        per_category = {c: _evaluate_from_cache(model, loader, metrics, args.device) for c, loader in test_loaders.items()}
        for report in per_category.values():
            report["mAD"] = float(np.mean(list(report.values())))
        mean_metrics = {m.name: float(np.mean([per_category[c][m.name] for c in categories])) for m in metrics}
        mean_metrics["mAD"] = float(np.mean(list(mean_metrics.values())))
        if logger is not None:
            log_dict = {f"{c}/{k}": v for c, r in per_category.items() for k, v in r.items()}
            log_dict.update({f"mean/{k}": v for k, v in mean_metrics.items()})
            log_dict["epoch"] = epoch
            logger.log(log_dict)
        return per_category, mean_metrics

    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        for enc_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            enc_batch = [f.to(args.device) for f in enc_batch]
            avg_loss += model.train_step_from_features(enc_batch, train_args)
        avg_loss /= len(train_loader)

        if train_args.scheduler is not None:
            train_args.scheduler.step()

        if logger is not None:
            logger.log({"epoch": epoch, "train_loss": avg_loss})
        print(f"[epoch {epoch}] train_loss={avg_loss:.5f}")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            _, mean_metrics = evaluate_all_categories(epoch)
            print(f"[epoch {epoch}] loss={avg_loss:.5f} mean={mean_metrics}")

    per_category, mean_metrics = evaluate_all_categories(args.epochs - 1)

    print("\nFinal per-category results:")
    for c in categories:
        print(f"  {c}: {per_category[c]}")
    print(f"Mean: {mean_metrics}")

    if args.save_path:
        model.save(args.save_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return per_category, mean_metrics
