"""
UniAD model for anomaly detection integrated into MoViAD.
Code adapted from:
    Title: Towards Unsupervised Anomaly Detection (UniAD)
    Authors: Zhiyuan You et al.
    URL: https://github.com/zhiyuanyou/UniAD
    License: MIT
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.uniad.components import UniADCore, MFCN

@dataclass
class UniADTrainArgs(TrainingArgs):
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    step_size: int = 800
    gamma: float = 0.1
    scheduler: object = None

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.uniad_core.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss()
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
            )


class UniAD(VADModel):

    def __init__(
        self,
        feature_extractor: CustomFeatureExtractor,
        input_size: Tuple[int, int] = (224, 224),
        feature_size: List[int] = [14, 14],
        hidden_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        pos_embed_type: str = "learned",
        neighbor_mask: Optional[dict] = None,
        feature_jitter: Optional[dict] = None,
        use_mfcn: bool = False,
        mfcn_instrides: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.input_size = input_size
        self.feature_size = feature_size
        self.use_mfcn = use_mfcn
        self.device = feature_extractor.device

        instride = self.input_size[0] // self.feature_size[0]

        inplanes = self._get_feature_channels()

        if use_mfcn and isinstance(inplanes, list) and mfcn_instrides is not None:
            self.neck = MFCN(
                inplanes=inplanes,
                instrides=mfcn_instrides,
                outstrides=[instride],
            ).to(self.device)
            transformer_inplanes = sum(inplanes)
        else:
            self.neck = None
            transformer_inplanes = inplanes if isinstance(inplanes, int) else inplanes[0]

        self.uniad_core = UniADCore(
            inplanes=[transformer_inplanes],
            instrides=[instride],
            feature_size=feature_size,
            hidden_dim=hidden_dim,
            pos_embed_type=pos_embed_type,
            initializer={"method": "xavier_uniform"},
            neighbor_mask=neighbor_mask,
            feature_jitter=feature_jitter,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        ).to(self.device)

    def _get_feature_channels(self):
        dummy = torch.zeros(1, 3, self.input_size[0], self.input_size[1])
        with torch.no_grad():
            features = self.feature_extractor(dummy.to(self.device))
        if isinstance(features, list):
            if len(features) == 1:
                return features[0].shape[1]
            return [f.shape[1] for f in features]
        else:
            vals = list(features.values())
            if len(vals) == 1:
                return vals[0].shape[1]
            return [v.shape[1] for v in vals]

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x.to(self.device))

        feat_list = features if isinstance(features, list) else list(features.values())

        if self.use_mfcn and self.neck is not None:
            feat = self.neck(feat_list)
        else:
            feat = feat_list[-1]

            feat = F.interpolate(
                feat,
                size=(self.feature_size[0], self.feature_size[1]),
                mode="bilinear",
                align_corners=False,
            )
        return feat

    def forward(self, x: torch.Tensor):
        feature_align = self._extract_features(x)
        feature_rec, pred = self.uniad_core(feature_align)

        if self.training:
            return feature_rec, feature_align
        else:
            anomaly_maps = F.interpolate(
                pred, size=self.input_size, mode="bilinear", align_corners=False
            )
            anomaly_scores = anomaly_maps.flatten(1).max(dim=1).values
            return anomaly_maps, anomaly_scores

    def train_step(self, batch: torch.Tensor, training_args: UniADTrainArgs):
        batch = batch.to(self.device)

        feature_rec, feature_align = self.forward(batch)
        loss = training_args.loss_function(feature_rec, feature_align)

        training_args.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.uniad_core.parameters(), max_norm=0.1)
        training_args.optimizer.step()

        return loss.item()

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")