"""
MambaAD model for anomaly detection integrated into MoViAD.
Code adapted from:
    Title: MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection
    Authors: Haoyang He et al.
    URL: https://github.com/lewandofskee/MambaAD
    License: (see original repository)
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

from moviad.models.components.rd4ad.resnet import resnet34
from moviad.models.mambaad.components import MambaDecoder
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel


@dataclass
class MambaADTrainArgs(TrainingArgs):
    lr: float = 5e-3
    weight_decay: float = 1e-4
    loss_weight: float = 5.0
    use_amp: bool = True

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                list(model.bn.parameters()) + list(model.decoder.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss()
        self.amp_enabled = self.use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)


class MambaAD(VADModel):

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        base_channels: int = 64,
        decoder_depths: List[int] = (3, 4, 6, 3),
        d_state: int = 16,
        num_direction: int = 8,
        use_checkpoint: bool = False,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.input_size = input_size
        self.device = torch.device("cpu")

        self.encoder, self.bn = resnet34(pretrained=True)
        self.decoder = MambaDecoder(
            base_channels=base_channels,
            bottleneck_size=input_size[0] // 32,
            depths=decoder_depths,
            d_state=d_state,
            num_direction=num_direction,
            use_checkpoint=use_checkpoint,
            chunk_size=chunk_size,
        )

    def to(self, device: torch.device):
        self.encoder.to(device)
        self.bn.to(device)
        self.decoder.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.bn.train(mode)
        self.decoder.train(mode)
        return self

    def _extract_features(self, batch: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            return self.encoder(batch.to(self.device))

    def forward(self, batch: torch.Tensor):
        enc_batch = self._extract_features(batch)
        return self.forward_from_features(enc_batch)

    def forward_from_features(self, enc_batch: List[torch.Tensor]):
        bn_batch = self.bn(enc_batch)
        dec_batch = self.decoder(bn_batch)

        if self.training:
            return enc_batch, dec_batch
        else:
            return self.post_process(enc_batch, dec_batch)

    def train_step(self, batch: torch.Tensor, training_args: MambaADTrainArgs):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        enc_batch = self._extract_features(batch)
        return self.train_step_from_features(enc_batch, training_args)

    def train_step_from_features(self, enc_batch: List[torch.Tensor], training_args: MambaADTrainArgs):
        device_type = "cuda" if training_args.amp_enabled else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=training_args.amp_enabled):
            enc_batch, dec_batch = self.forward_from_features(enc_batch)

            loss = 0
            for i in range(len(dec_batch)):
                loss += training_args.loss_function(dec_batch[i], enc_batch[i])
            loss = loss * training_args.loss_weight

        training_args.optimizer.zero_grad()
        training_args.scaler.scale(loss).backward()
        training_args.scaler.unscale_(training_args.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        training_args.scaler.step(training_args.optimizer)
        training_args.scaler.update()

        return loss.item()

    def post_process(self, enc_batch, dec_batch) -> torch.Tensor:
        anomaly_map = None
        sigma = 4
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

        for i in range(len(enc_batch)):
            fs = dec_batch[i]
            ft = enc_batch[i]

            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(
                a_map, size=self.input_size, mode="bilinear", align_corners=True,
            )

            anomaly_map = a_map if anomaly_map is None else anomaly_map + a_map

        anomaly_map = blur(anomaly_map)
        anomaly_scores = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]
        return anomaly_map, anomaly_scores

    def train_epoch(self, epoch, train_dataloader, training_args: MambaADTrainArgs):
        avg_batch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}]"):
            avg_batch_loss += self.train_step(batch, training_args)
        avg_batch_loss /= len(train_dataloader)

        if training_args.scheduler is not None:
            training_args.scheduler.step()

        return avg_batch_loss

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")
