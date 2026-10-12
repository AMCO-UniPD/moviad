from __future__ import annotations

from typing import Callable
from tqdm import tqdm
import torch
import numpy as np

from .metrics import MetricLvl, Metric
from moviad.models.vad_model import VADModel


def min_max_norm(x):
    values = np.asarray(x)
    value_range = values.max() - values.min()
    if value_range == 0:
        return np.zeros_like(values, dtype=np.result_type(values.dtype, np.float32))
    return (values - values.min()) / value_range


class Evaluator:
    """
    This class will evaluate the trained model on the test set
    and it will produce the evaluation metrics needed

    Args:
        test_dataloader (Dataloader): test dataloader
        device (torch.device): device where to run the model
    """

    @staticmethod
    def evaluate(model: VADModel, dataloader, metrics: list[Metric], device, postprocess: Callable = min_max_norm) -> dict:
        """
        Args:
            model: a model object on which you can call model.predict(batched_images)
                and returns a tuple of anomaly_maps and anomaly_scores
            postprocess (Callable): a function to postprocess the predicted anomaly maps
        Returns:
            dict: a dictionary with metric names as keys and computed metric values as values
        """
        model.eval()

        gt_mask, gt_label, pred_anom_map, pred_anom_score = [], [], [], []

        for image, label, mask, path in tqdm(dataloader, desc="Eval"):
            with torch.no_grad():
                anom_maps, anom_scores = model(image.to(device))
            gt_mask.append(mask.cpu().numpy().astype(int))
            gt_label.append(label.cpu().numpy())
            pred_anom_map.append(anom_maps.cpu().numpy())
            pred_anom_score.append(anom_scores.cpu().numpy())

        gt_mask = np.concatenate(gt_mask)
        gt_label = np.concatenate(gt_label)
        pred_anom_map = postprocess(np.concatenate(pred_anom_map))
        pred_anom_score = np.concatenate(pred_anom_score)

        report = {}
        for metric in metrics:
            if metric.level == MetricLvl.IMAGE:
                gt, pred = gt_label, pred_anom_score
            else:
                gt, pred = gt_mask, pred_anom_map

            report[metric.name] = metric.compute(gt, pred)

        return report
