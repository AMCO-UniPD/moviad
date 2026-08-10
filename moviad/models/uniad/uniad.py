import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

#Add the uniad directory to path so "uniad_models" can be found
UNIAD_DIR = os.path.dirname(__file__)
if UNIAD_DIR not in sys.path:
    sys.path.insert(0, UNIAD_DIR)

#Add moviad root to path for accessing utilities
MOVIAD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if MOVIAD_DIR not in sys.path:
    sys.path.insert(0, MOVIAD_DIR)

from uniad_models.reconstructions.uniad import UniAD as UniADOriginal
from utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


class UniAD(nn.Module):
    """
    MoViAD wrapper for UniAD (NeurIPS 2022).

    Code adapted from:
        Title: Towards Unsupervised Anomaly Detection (UniAD)
        Authors: Zhiyuan You et al.
        URL: https://github.com/zhiyuanyou/UniAD
        License: MIT
    """

    def __init__(
        self,
        device: torch.device,
        input_size: tuple,
        feature_extractor: CustomFeatureExtractor,
        feature_size: list = [16, 16],
        hidden_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pos_embed_type: str = "learned",
    ):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size

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
        feat = F.interpolate(
            feat,
            size=(self.feature_size[0], self.feature_size[1]),
            mode="bilinear",
            align_corners=False,
        )
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
            anomaly_maps = F.interpolate(
                pred,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            anomaly_scores = anomaly_maps.flatten(1).max(dim=1).values
            return anomaly_maps, anomaly_scores

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")