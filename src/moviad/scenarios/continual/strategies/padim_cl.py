"""PaDiM continual learning strategies.

PadimCLUnimodal: Incremental update of a single Gaussian per spatial location
    using sufficient statistics (sum, sum of outer products, count).

PadimCLMultimodal: Stores one Gaussian per task, takes minimum anomaly score
    across all task Gaussians at inference. Includes classification sanity check.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from tqdm import tqdm

from moviad.datasets.vad_dataset import VADDataset
from moviad.models.padim.padim import Padim, PadimTrainArgs
from moviad.models.padim.padim_lite import PadimLite
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.utilities.evaluation.metrics import Metric


def _extract_embeddings(
    padim: Padim,
    train_dataset: VADDataset,
    batch_size: int,
    device,
    num_workers: int = 4,
):
    """Extract embedding vectors from training data."""
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    layer_outputs = {layer: [] for layer in padim.layers_idxs}

    padim.train()
    for batch in tqdm(train_dataloader, desc="| feature extraction | train |"):
        batch = batch.to(device)
        outputs = padim.forward(batch)
        for layer, output in outputs.items():
            layer_outputs[layer].extend(output)

    return padim.raw_feature_maps_to_embeddings(layer_outputs)


def _compute_sufficient_stats(embedding_vectors: torch.Tensor):
    """Compute sufficient statistics (n, s1, s2) per spatial location.

    Args:
        embedding_vectors: (B, C, H, W) tensor

    Returns:
        n: int, number of samples
        s1: (C, H*W) sum of features
        s2: (C, C, H*W) sum of outer products
    """
    B, C, H, W = embedding_vectors.size()
    X = embedding_vectors.view(B, C, H * W).cpu().numpy()  # (B, C, H*W)

    n = B
    s1 = X.sum(axis=0)  # (C, H*W)
    # s2[c1, c2, hw] = sum over batch of X[b, c1, hw] * X[b, c2, hw]
    s2 = np.einsum('bci,bdi->cdi', X, X)  # (C, C, H*W)

    return n, s1, s2


def _stats_to_gaussian(n, s1, s2, reg=0.01):
    """Convert sufficient statistics to mean and covariance.

    Returns:
        mean: (C, H*W)
        cov: (C, C, H*W)
    """
    C = s1.shape[0]
    mean = s1 / n  # (C, H*W)
    # cov = E[X^2] - E[X]^2 + regularization
    cov = s2 / n - np.einsum('ci,di->cdi', mean, mean)  # (C, C, H*W)
    I = np.identity(C)
    cov += reg * I[:, :, np.newaxis]
    return mean, cov


class PadimCLUnimodal(ContinualModel):
    """Incremental Gaussian update using sufficient statistics.

    Maintains running sums (s1, s2, n) across tasks, fitting a single
    unimodal Gaussian that incrementally incorporates new task data.
    """

    def __init__(self, padim_model: Padim):
        super().__init__(padim_model)
        self.n = 0
        self.s1 = None  # (C, H*W)
        self.s2 = None  # (C, C, H*W)

    def start_task(self, task_index=None, train_dataset=None, train_args=None):
        pass

    def train_task(self, task_index, train_dataset, eval_dataset,
                   metrics, device, logger=None, train_args=None):
        padim: Padim = self.vad_model
        batch_size = train_args.batch_size if train_args else 32
        num_workers = train_args.num_workers if train_args else 4

        embeddings = _extract_embeddings(
            padim, train_dataset, batch_size, device, num_workers
        )
        n_new, s1_new, s2_new = _compute_sufficient_stats(embeddings)

        # Incremental update
        if self.s1 is None:
            self.n, self.s1, self.s2 = n_new, s1_new, s2_new
        else:
            self.n += n_new
            self.s1 += s1_new
            self.s2 += s2_new

        # Update PaDiM model parameters
        mean, cov = _stats_to_gaussian(self.n, self.s1, self.s2)
        padim.gauss_mean = mean
        padim.gauss_cov = cov

    def end_task(self, task_index=None, train_dataset=None):
        pass


class PadimCLMultimodal(ContinualModel):
    """Mixture of Gaussians: one Gaussian per task.

    Stores per-task Gaussian parameters. At inference, computes anomaly
    score from each task's Gaussian and takes the minimum.
    Includes classification: argmin over task anomaly scores.
    """

    def __init__(self, padim_model: Padim):
        super().__init__(padim_model)
        # task_id -> (mean, cov) where mean: (C, H*W), cov: (C, C, H*W)
        self.task_gaussians = {}

    def start_task(self, task_index=None, train_dataset=None, train_args=None):
        pass

    def train_task(self, task_index, train_dataset, eval_dataset,
                   metrics, device, logger=None, train_args=None):
        padim: Padim = self.vad_model
        batch_size = train_args.batch_size if train_args else 32
        num_workers = train_args.num_workers if train_args else 4

        embeddings = _extract_embeddings(
            padim, train_dataset, batch_size, device, num_workers
        )
        n, s1, s2 = _compute_sufficient_stats(embeddings)
        mean, cov = _stats_to_gaussian(n, s1, s2)

        self.task_gaussians[task_index] = (mean, cov)

        # Set the latest task's Gaussian as the active one (for compatibility)
        padim.gauss_mean = mean
        padim.gauss_cov = cov

    def end_task(self, task_index=None, train_dataset=None):
        pass

    def _extract_per_task_distances(self, batch: torch.Tensor):
        """Extract raw Mahalanobis distances for each task (before upsampling).

        Returns:
            embedding_vectors: (B, C, H_emb, W_emb)
            task_dist_lists: dict {task_id: (B, H_emb, W_emb) Mahalanobis distances}
        """
        padim: Padim = self.vad_model
        padim.eval()

        layer_outputs = {layer: [] for layer in padim.layers_idxs}
        with torch.no_grad():
            padim.backbone(batch.to(padim.device))
        for layer, output in zip(padim.layers_idxs, padim.outputs):
            layer_outputs[layer].append(output.cpu().detach())
        padim.outputs = []

        embedding_vectors = padim.raw_feature_maps_to_embeddings(layer_outputs)

        task_dist_lists = {}
        for task_id in sorted(self.task_gaussians.keys()):
            mean, cov = self.task_gaussians[task_id]
            padim.gauss_mean = mean
            padim.gauss_cov = cov
            task_dist_lists[task_id] = padim.compute_distances(embedding_vectors)

        return embedding_vectors, task_dist_lists

    def forward(self, batch: torch.Tensor):
        """Classify via joint likelihood across all patches, then anomaly score from best task."""
        padim: Padim = self.vad_model

        _, task_dist_lists = self._extract_per_task_distances(batch)

        # Classification: argmin of mean squared Mahalanobis distance across ALL patches
        # This is proportional to -log p(image | task) under independent patch Gaussians
        all_mean_d2 = []  # (n_tasks, B) — mean d² across patches for classification
        all_score_maps = []
        all_img_scores = []

        for task_id in sorted(task_dist_lists.keys()):
            dist_list = task_dist_lists[task_id]  # (B, H_emb, W_emb)

            # Mean of squared distances across all patches → classification score
            B = dist_list.shape[0]
            mean_d2 = (dist_list.numpy() ** 2).reshape(B, -1).mean(axis=1)  # (B,)
            all_mean_d2.append(mean_d2)

            # Upsample + smooth → anomaly score map
            score_map = (
                F.interpolate(
                    dist_list.unsqueeze(1),
                    size=batch.size(2),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .numpy()
            )
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)

            img_scores = score_map.reshape(B, -1).max(axis=1)  # anomaly score = max

            all_score_maps.append(score_map)
            all_img_scores.append(img_scores)

        all_mean_d2 = np.stack(all_mean_d2, axis=0)  # (n_tasks, B)
        all_score_maps = np.stack(all_score_maps, axis=0)
        all_img_scores = np.stack(all_img_scores, axis=0)

        # Classification: pick task with lowest mean d² (= highest joint likelihood)
        best_task = np.argmin(all_mean_d2, axis=0)  # (B,)
        batch_idx = np.arange(batch.size(0))

        # Anomaly output from the classified task
        min_score_maps = all_score_maps[best_task, batch_idx]
        min_img_scores = all_img_scores[best_task, batch_idx]
        min_score_maps = np.expand_dims(min_score_maps, axis=1)

        return torch.from_numpy(min_score_maps), torch.from_numpy(min_img_scores)

    def classify(self, batch: torch.Tensor) -> np.ndarray:
        """Return predicted task index for each image in the batch.

        Uses argmin of mean squared Mahalanobis distance across all patches
        (= argmax joint log-likelihood under independent patch Gaussians).
        """
        _, task_dist_lists = self._extract_per_task_distances(batch)

        all_mean_d2 = []
        for task_id in sorted(task_dist_lists.keys()):
            dist = task_dist_lists[task_id].numpy()  # (B, H, W)
            B = dist.shape[0]
            mean_d2 = (dist ** 2).reshape(B, -1).mean(axis=1)
            all_mean_d2.append(mean_d2)

        all_mean_d2 = np.stack(all_mean_d2, axis=0)  # (n_tasks, B)
        return np.argmin(all_mean_d2, axis=0)  # (B,)


# ---- PadimLite CL strategies (diagonal covariance) ----

def _compute_diagonal_sufficient_stats(embedding_vectors: torch.Tensor):
    """Compute diagonal sufficient statistics (n, s1, s2_diag) per spatial location.

    Args:
        embedding_vectors: (B, C, H, W) tensor

    Returns:
        n: int, number of samples
        s1: (C, H*W) sum of features
        s2_diag: (C, H*W) sum of squared features (per-channel)
    """
    B, C, H, W = embedding_vectors.size()
    X = embedding_vectors.view(B, C, H * W).cpu().numpy()  # (B, C, H*W)

    n = B
    s1 = X.sum(axis=0)  # (C, H*W)
    s2_diag = (X ** 2).sum(axis=0)  # (C, H*W)

    return n, s1, s2_diag


def _diag_stats_to_gaussian(n, s1, s2_diag, reg=0.01):
    """Convert diagonal sufficient statistics to mean and variance.

    Returns:
        mean: (C, H*W)
        var: (C, H*W) — diagonal covariance (per-channel variance)
    """
    mean = s1 / n  # (C, H*W)
    var = s2_diag / n - mean ** 2 + reg  # (C, H*W)
    return mean, var


class PadimLiteCLUnimodal(ContinualModel):
    """Incremental diagonal Gaussian update using sufficient statistics."""

    def __init__(self, padim_lite_model: PadimLite):
        super().__init__(padim_lite_model)
        self.n = 0
        self.s1 = None  # (C, H*W)
        self.s2_diag = None  # (C, H*W)

    def start_task(self, task_index=None, train_dataset=None, train_args=None):
        pass

    def train_task(self, task_index, train_dataset, eval_dataset,
                   metrics, device, logger=None, train_args=None):
        padim: PadimLite = self.vad_model
        batch_size = train_args.batch_size if train_args else 32
        num_workers = train_args.num_workers if train_args else 4

        embeddings = _extract_embeddings(
            padim, train_dataset, batch_size, device, num_workers
        )
        n_new, s1_new, s2_diag_new = _compute_diagonal_sufficient_stats(embeddings)

        if self.s1 is None:
            self.n, self.s1, self.s2_diag = n_new, s1_new, s2_diag_new
        else:
            self.n += n_new
            self.s1 += s1_new
            self.s2_diag += s2_diag_new

        mean, var = _diag_stats_to_gaussian(self.n, self.s1, self.s2_diag)
        padim.gauss_mean = mean
        padim.diagonal_gauss_cov = var

    def end_task(self, task_index=None, train_dataset=None):
        pass


class PadimLiteCLMultimodal(ContinualModel):
    """Mixture of diagonal Gaussians: one per task, min anomaly score inference."""

    def __init__(self, padim_lite_model: PadimLite):
        super().__init__(padim_lite_model)
        # task_id -> (mean, var) where mean: (C, H*W), var: (C, H*W)
        self.task_gaussians = {}

    def start_task(self, task_index=None, train_dataset=None, train_args=None):
        pass

    def train_task(self, task_index, train_dataset, eval_dataset,
                   metrics, device, logger=None, train_args=None):
        padim: PadimLite = self.vad_model
        batch_size = train_args.batch_size if train_args else 32
        num_workers = train_args.num_workers if train_args else 4

        embeddings = _extract_embeddings(
            padim, train_dataset, batch_size, device, num_workers
        )
        n, s1, s2_diag = _compute_diagonal_sufficient_stats(embeddings)
        mean, var = _diag_stats_to_gaussian(n, s1, s2_diag)

        self.task_gaussians[task_index] = (mean, var)
        padim.gauss_mean = mean
        padim.diagonal_gauss_cov = var

    def end_task(self, task_index=None, train_dataset=None):
        pass

    def _extract_per_task_distances(self, batch: torch.Tensor):
        """Extract raw Mahalanobis distances for each task."""
        padim: PadimLite = self.vad_model
        padim.eval()

        layer_outputs = {layer: [] for layer in padim.layers_idxs}
        with torch.no_grad():
            padim.backbone(batch.to(padim.device))
        for layer, output in zip(padim.layers_idxs, padim.outputs):
            layer_outputs[layer].append(output.cpu().detach())
        padim.outputs = []

        embedding_vectors = padim.raw_feature_maps_to_embeddings(layer_outputs)

        task_dist_lists = {}
        for task_id in sorted(self.task_gaussians.keys()):
            mean, var = self.task_gaussians[task_id]
            padim.gauss_mean = mean
            padim.diagonal_gauss_cov = var
            task_dist_lists[task_id] = padim.compute_distances(embedding_vectors)

        return embedding_vectors, task_dist_lists

    def forward(self, batch: torch.Tensor):
        """Classify via joint likelihood across all patches, then anomaly score from best task."""
        _, task_dist_lists = self._extract_per_task_distances(batch)

        all_mean_d2 = []
        all_score_maps = []
        all_img_scores = []

        for task_id in sorted(task_dist_lists.keys()):
            dist_list = task_dist_lists[task_id]
            B = dist_list.shape[0]

            mean_d2 = (dist_list.numpy() ** 2).reshape(B, -1).mean(axis=1)
            all_mean_d2.append(mean_d2)

            score_map = (
                F.interpolate(
                    dist_list.unsqueeze(1),
                    size=batch.size(2),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .numpy()
            )
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)

            img_scores = score_map.reshape(B, -1).max(axis=1)
            all_score_maps.append(score_map)
            all_img_scores.append(img_scores)

        all_mean_d2 = np.stack(all_mean_d2, axis=0)
        all_score_maps = np.stack(all_score_maps, axis=0)
        all_img_scores = np.stack(all_img_scores, axis=0)

        best_task = np.argmin(all_mean_d2, axis=0)
        batch_idx = np.arange(batch.size(0))

        min_score_maps = all_score_maps[best_task, batch_idx]
        min_img_scores = all_img_scores[best_task, batch_idx]
        min_score_maps = np.expand_dims(min_score_maps, axis=1)

        return torch.from_numpy(min_score_maps), torch.from_numpy(min_img_scores)

    def classify(self, batch: torch.Tensor) -> np.ndarray:
        """Return predicted task index using joint likelihood across all patches."""
        _, task_dist_lists = self._extract_per_task_distances(batch)

        all_mean_d2 = []
        for task_id in sorted(task_dist_lists.keys()):
            dist = task_dist_lists[task_id].numpy()
            B = dist.shape[0]
            mean_d2 = (dist ** 2).reshape(B, -1).mean(axis=1)
            all_mean_d2.append(mean_d2)

        all_mean_d2 = np.stack(all_mean_d2, axis=0)
        return np.argmin(all_mean_d2, axis=0)
