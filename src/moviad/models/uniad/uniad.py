"""UniAD model for anomaly detection integrated into MoViAD.

Code adapted from:
    Title: Towards Unsupervised Anomaly Detection (UniAD)
    Authors: Zhiyuan You et al.
    URL: https://github.com/zhiyuanyou/UniAD
    License: MIT
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm

# Add uniad directory to path so "uniad_models" can be found
UNIAD_DIR = os.path.dirname(__file__)
if UNIAD_DIR not in sys.path:
    sys.path.insert(0, UNIAD_DIR)

from uniad_models.reconstructions.uniad import UniAD as UniADOriginal
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


@dataclass
class UniADTrainArgs(TrainingArgs):
    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.uniad.parameters(), lr=1e-4
            )
        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss()


class UniAD(VADModel):
    """
    MoViAD integration of UniAD follows the VADModel interface of the continual branch.
    """
    def __init__(
        self,
        feature_extractor: CustomFeatureExtractor,
        feature_size: list = [16, 16],
        hidden_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pos_embed_type: str = "learned",
        input_size: tuple = (224, 224),
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size
        self.input_size = input_size
        self.device = feature_extractor.device

        inplanes = self._get_feature_channels()
        print(f"Backbone feature channels: {inplanes}")

        self.uniad = UniADOriginal(
            inplanes=[inplanes],
            instrides=[1],
            feature_size=feature_size,
            feature_jitter=None,
            neighbor_mask=None,
            hidden_dim=hidden_dim,
            pos_embed_type=pos_embed_type,
            save_recon=None,
            initializer={"method": "xavier_uniform"},
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)

    def _get_feature_channels(self):
        dummy = torch.zeros(1, 3, self.input_size[0], self.input_size[1])
        with torch.no_grad():
            features = self.feature_extractor(dummy.to(self.device))
        feat = features[-1] if isinstance(features, list) else list(features.values())[-1]
        return feat.shape[1]

    def _extract_and_resize(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x.to(self.device))
        feat = features[-1] if isinstance(features, list) else list(features.values())[-1]
        feat = F.interpolate(feat, size=(self.feature_size[0], self.feature_size[1]),
                             mode="bilinear", align_corners=False)
        return feat

    def forward(self, x: torch.Tensor):
        feature_align = self._extract_and_resize(x)
        uniad_input = {
            "feature_align": feature_align,
            "clsname": [""] * x.shape[0],
            "filename": [""] * x.shape[0],
        }
        output = self.uniad(uniad_input)

        if self.training:
            return output
        else:
            pred = output["pred"]
            anomaly_maps = F.interpolate(pred, size=self.input_size,
                                         mode="bilinear", align_corners=False)
            anomaly_scores = anomaly_maps.flatten(1).max(dim=1).values
            return anomaly_maps, anomaly_scores

    def train_step(self, batch: torch.Tensor, training_args: UniADTrainArgs):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(self.device)

        output = self.forward(batch)
        loss = training_args.loss_function(output["feature_rec"], output["feature_align"])

        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()

        return loss.item()

    def train_epoch(self, epoch, train_dataloader, training_args: UniADTrainArgs):
        avg_batch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}]"):
            avg_batch_loss += self.train_step(batch, training_args)
        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")
