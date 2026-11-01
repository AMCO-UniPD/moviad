"""
DiAD model for anomaly detection integrated into MoViAD.
Code adapted from:
    Title: DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection
    Authors: Haoyang He et al.
    URL: https://github.com/lewandofskee/DiAD
    License: Apache-2.0
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, UNet2DConditionModel
from tqdm import tqdm

from moviad.models.diad.components import SemanticGuidedNetwork
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


@dataclass
class DiADConfig:
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    cross_attention_dim: int = 768
    attention_head_dim: int = 8
    text_tokens: int = 77
    vae_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    latent_channels: int = 4
    scale_factor: float = 0.18215
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    ddim_steps: int = 10
    start_timestep: int = 999
    only_mid_control: bool = True
    train_unet_decoder: bool = True
    train_middle: bool = False
    use_checkpoint: bool = True
    feature_extractor_name: str = "resnet50"
    feature_nodes: Tuple[str, ...] = ("layer1", "layer2", "layer3")
    pretrained_extractor: bool = True
    gaussian_sigma: float = 5.0
    pool_kernel: int = 8
    pool_rounds: int = 8


@dataclass
class DiADTrainArgs(TrainingArgs):
    lr: float = 1e-5
    weight_decay: float = 1e-2
    accumulate_grad_batches: int = 4
    step_count: int = field(default=0, init=False)

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.trainable_parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss()
        self.step_count = 0


def _symmetric_pad(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    head = x.narrow(dim, 0, pad).flip(dim)
    tail = x.narrow(dim, x.shape[dim] - pad, pad).flip(dim)
    return torch.cat([head, x, tail], dim=dim)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(4.0 * sigma + 0.5)
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    x = _symmetric_pad(x, radius, dim=3)
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = _symmetric_pad(x, radius, dim=2)
    return F.conv2d(x, kernel.view(1, 1, -1, 1))


class DiAD(VADModel):

    def __init__(
        self,
        config: Optional[DiADConfig] = None,
        feature_extractor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config or DiADConfig()
        cfg = self.config

        self.unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=cfg.latent_channels,
            out_channels=cfg.latent_channels,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
            cross_attention_dim=cfg.cross_attention_dim,
            attention_head_dim=cfg.attention_head_dim,
        )
        self.sg = SemanticGuidedNetwork(
            ControlNetModel.from_unet(self.unet, conditioning_channels=3),
            cfg.block_out_channels,
            only_mid_control=cfg.only_mid_control,
        )
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=cfg.latent_channels,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=cfg.vae_block_out_channels,
            layers_per_block=cfg.layers_per_block,
        )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            timestep_spacing="leading",
        )
        self.register_buffer("text_embedding", torch.zeros(1, cfg.text_tokens, cfg.cross_attention_dim))
        self.feature_extractor = feature_extractor if feature_extractor is not None else self._build_feature_extractor()

        self._freeze()
        if cfg.use_checkpoint:
            self.unet.enable_gradient_checkpointing()
            self.sg.controlnet.enable_gradient_checkpointing()

    def _build_feature_extractor(self) -> nn.Module:
        extractor = CustomFeatureExtractor(
            self.config.feature_extractor_name,
            list(self.config.feature_nodes),
            torch.device("cpu"),
            self.config.pretrained_extractor,
        )
        return extractor.model

    def _extract_features(self, image: torch.Tensor) -> List[torch.Tensor]:
        features = self.feature_extractor(image)
        if isinstance(features, dict):
            features = list(features.values())
        return features

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        trainable = list(self.sg.parameters())
        if self.config.train_unet_decoder:
            trainable += list(self.unet.up_blocks.parameters())
            trainable += list(self.unet.conv_norm_out.parameters())
            trainable += list(self.unet.conv_out.parameters())
        if self.config.train_middle:
            trainable += list(self.unet.mid_block.parameters())
        for p in trainable:
            p.requires_grad_(True)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def device(self) -> torch.device:
        return self.text_embedding.device

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        self.feature_extractor.eval()
        return self

    def init_sg_from_unet(self):
        self.sg.controlnet.load_state_dict(self.unet.state_dict(), strict=False)

    def set_text_embedding(self, embedding: torch.Tensor):
        self.text_embedding.copy_(embedding.reshape_as(self.text_embedding))

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(image).latent_dist.sample() * self.config.scale_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.config.scale_factor).sample

    def predict_noise(self, noisy_latent: torch.Tensor, timestep: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        text = self.text_embedding.expand(noisy_latent.shape[0], -1, -1)
        down, mid = self.sg(noisy_latent, timestep, text, image)
        return self.unet(
            noisy_latent,
            timestep,
            encoder_hidden_states=text,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
        ).sample

    @torch.no_grad()
    def reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = image.shape[0]
        latent = self.encode(image)
        start = torch.full((batch,), cfg.start_timestep, device=image.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(latent, torch.randn_like(latent), start)
        self.scheduler.set_timesteps(cfg.ddim_steps, device=image.device)
        for t in self.scheduler.timesteps:
            noise_pred = self.predict_noise(noisy, t.expand(batch), image)
            noisy = self.scheduler.step(noise_pred, t, noisy).prev_sample
        return self.decode(noisy)

    @torch.no_grad()
    def post_process(self, image: torch.Tensor, reconstruction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        features_in = self._extract_features(image)
        features_out = self._extract_features(reconstruction)

        anomaly_map = 0
        for f_in, f_out in zip(features_in, features_out):
            a_map = 1 - F.cosine_similarity(f_in, f_out)
            a_map = F.interpolate(a_map.unsqueeze(1), size=image.shape[-2:], mode="bilinear", align_corners=True)
            anomaly_map = anomaly_map + a_map

        anomaly_map = gaussian_blur(anomaly_map, cfg.gaussian_sigma)

        pooled = anomaly_map
        for _ in range(cfg.pool_rounds):
            pooled = F.avg_pool2d(pooled, cfg.pool_kernel, stride=1)
        anomaly_scores = pooled.flatten(1).max(dim=1)[0]
        return anomaly_map, anomaly_scores

    def forward(self, batch: torch.Tensor):
        image = batch.to(self.device)
        return self.post_process(image, self.reconstruct(image))

    def train_step(self, batch: torch.Tensor, training_args: DiADTrainArgs):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        image = batch.to(self.device)

        latent = self.encode(image)
        noise = torch.randn_like(latent)
        timestep = torch.randint(0, self.config.num_train_timesteps, (image.shape[0],), device=self.device)
        noisy = self.scheduler.add_noise(latent, noise, timestep)

        loss = training_args.loss_function(self.predict_noise(noisy, timestep, image), noise)
        (loss / training_args.accumulate_grad_batches).backward()

        training_args.step_count += 1
        if training_args.step_count % training_args.accumulate_grad_batches == 0:
            training_args.optimizer.step()
            training_args.optimizer.zero_grad()

        return loss.item()

    def train_epoch(self, epoch, train_dataloader, training_args: DiADTrainArgs):
        training_args.step_count = 0
        avg_batch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}]"):
            avg_batch_loss += self.train_step(batch, training_args)
        avg_batch_loss /= len(train_dataloader)

        if training_args.step_count % training_args.accumulate_grad_batches != 0:
            training_args.optimizer.step()
            training_args.optimizer.zero_grad()

        if training_args.scheduler is not None:
            training_args.scheduler.step()

        return avg_batch_loss

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")

    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer):
        trainable = {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}
        temporary_path = f"{path}.tmp"
        torch.save({"epoch": epoch, "trainable": trainable, "optimizer": optimizer.state_dict()}, temporary_path)
        os.replace(temporary_path, path)
        print(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer) -> int:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _, unexpected = self.load_state_dict(checkpoint["trainable"], strict=False)
        if unexpected:
            raise KeyError(f"Checkpoint does not match the model: unexpected={unexpected[:5]}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Checkpoint loaded from: {path}")
        return checkpoint["epoch"] + 1
