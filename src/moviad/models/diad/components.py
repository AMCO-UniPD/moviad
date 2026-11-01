"""
DiAD components for anomaly detection integrated into MoViAD.
Code adapted from:
    Title: DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection
    Authors: Haoyang He et al.
    URL: https://github.com/lewandofskee/DiAD
    License: Apache-2.0
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from diffusers import ControlNetModel


class SFF(nn.Module):

    def __init__(self, mid_channels: int, high_channels: int, out_channels: int):
        super().__init__()
        in_channels = {1: mid_channels, 2: high_channels, 3: high_channels}
        for src in (1, 2, 3):
            for dst in (1, 2, 3):
                setattr(self, f"down{src}{dst}", self._fusion_block(in_channels[src], out_channels))
        self.activation = nn.SiLU()

    @staticmethod
    def _fusion_block(in_channels: int, out_channels: int) -> nn.Sequential:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        nn.init.zeros_(conv.weight)
        return nn.Sequential(conv, nn.InstanceNorm2d(out_channels), nn.SiLU())

    def forward(self, encoder_outputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for dst in (1, 2, 3):
            h = encoder_outputs[8 + dst]
            for src in (1, 2, 3):
                h = h + getattr(self, f"down{src}{dst}")(encoder_outputs[5 + src])
            fused.append(self.activation(h))
        return fused


class SemanticGuidedNetwork(nn.Module):

    def __init__(
        self,
        controlnet: ControlNetModel,
        block_out_channels: Sequence[int],
        only_mid_control: bool = True,
    ):
        super().__init__()
        self.controlnet = controlnet
        self.sff = SFF(block_out_channels[1], block_out_channels[2], block_out_channels[3])
        self.only_mid_control = only_mid_control

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embedding: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        down, mid = self.controlnet(
            noisy_latent,
            timestep,
            encoder_hidden_states=text_embedding,
            controlnet_cond=image,
            return_dict=False,
        )
        down = list(down)
        down[9:12] = self.sff(down)
        if self.only_mid_control:
            for i in range(9):
                down[i] = torch.zeros_like(down[i])
        return tuple(down), mid
