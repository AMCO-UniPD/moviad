"""
MambaAD components for anomaly detection integrated into MoViAD.
Code adapted from:
    Title: MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection
    Authors: Haoyang He et al.
    URL: https://github.com/lewandofskee/MambaAD
    License: (see original repository)
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def _hilbert_xy_from_d(order: int, d: int):
    x, y = 0, 0
    t = d
    s = 1
    while s < (1 << order):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s <<= 1
    return x, y


def hilbert_order(size: int) -> torch.LongTensor:
    order = int(math.log2(size))
    assert 2 ** order == size, "Hilbert scan requires a power-of-two feature map size"
    coords = [_hilbert_xy_from_d(order, d) for d in range(size * size)]
    flat_index = [x * size + y for x, y in coords]
    return torch.tensor(flat_index, dtype=torch.long)


class SelectiveScan(nn.Module):

    def __init__(self, chunk_size: int = 256):
        super().__init__()

    def forward(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        return selective_scan_fn(x, dt.to(x.dtype), A.float(), B.to(x.dtype), C.to(x.dtype), D.float())


class SS2D(nn.Module):

    def __init__(self, d_model: int, size: int, d_state: int = 16, d_conv: int = 3,
                 expand: int = 2, num_direction: int = 8, dt_rank: int = None,
                 dt_min: float = 0.001, dt_max: float = 0.1, dt_init_floor: float = 1e-4,
                 chunk_size: int = 256):
        super().__init__()
        assert num_direction in (2, 4, 8)
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.num_direction = num_direction
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv // 2, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()

        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
            for _ in range(num_direction)
        ])
        self.dt_proj = nn.ModuleList([
            self._dt_init(self.dt_rank, self.d_inner, dt_min, dt_max, dt_init_floor)
            for _ in range(num_direction)
        ])

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A).unsqueeze(0).repeat(num_direction, 1, 1))
        self.Ds = nn.Parameter(torch.ones(num_direction, self.d_inner))

        self.selective_scan = SelectiveScan(chunk_size=chunk_size)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        order = hilbert_order(size)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(size * size)
        self.register_buffer("_scan_order", order)
        self.register_buffer("_scan_order_inv", inv_order)

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_min, dt_max, dt_init_floor):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    def _scan_encode(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        return x.reshape(B, D, H * W).index_select(-1, self._scan_order)

    def _scan_decode(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, D, L = x.shape
        return x.index_select(-1, self._scan_order_inv).reshape(B, D, H, W)

    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        L = H * W
        K = self.num_direction

        variants = [x]
        if K >= 4:
            variants.append(x.transpose(2, 3))
        if K >= 8:
            variants.append(torch.rot90(x, k=1, dims=(2, 3)))
            variants.append(torch.rot90(x, k=1, dims=(2, 3)).transpose(2, 3))

        scanned = torch.stack([self._scan_encode(v.contiguous()) for v in variants], dim=1)
        xs = torch.cat([scanned, torch.flip(scanned, dims=[-1])], dim=1)

        ys = []
        for k in range(K):
            xk = xs[:, k]
            params = self.x_proj[k](xk.transpose(1, 2))
            dt, Bp, Cp = torch.split(params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj[k](dt)).transpose(1, 2)
            Bp = Bp.transpose(1, 2)
            Cp = Cp.transpose(1, 2)
            A = -torch.exp(self.A_logs[k])
            ys.append(self.selective_scan(xk, dt, A, Bp, Cp, self.Ds[k]))
        ys = torch.stack(ys, dim=1)

        num_base = K // 2
        inv_y = torch.flip(ys[:, num_base:K], dims=[-1])

        outputs = []
        outputs.append(self._scan_decode(ys[:, 0], H, W))
        outputs.append(self._scan_decode(inv_y[:, 0], H, W))
        if K >= 4:
            outputs.append(self._scan_decode(ys[:, 1], W, H).transpose(2, 3))
            outputs.append(self._scan_decode(inv_y[:, 1], W, H).transpose(2, 3))
        if K >= 8:
            outputs.append(torch.rot90(self._scan_decode(ys[:, 2], H, W), k=3, dims=(2, 3)))
            outputs.append(torch.rot90(self._scan_decode(inv_y[:, 2], H, W), k=3, dims=(2, 3)))
            outputs.append(torch.rot90(self._scan_decode(ys[:, 3], W, H).transpose(2, 3), k=3, dims=(2, 3)))
            outputs.append(torch.rot90(self._scan_decode(inv_y[:, 3], W, H).transpose(2, 3), k=3, dims=(2, 3)))

        return sum(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_in = self.act(self.conv2d(x_in))

        y = self.forward_core(x_in)
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)
        y = y * self.act(z)
        out = self.out_proj(y)
        return out


class HSSBlock(nn.Module):

    def __init__(self, hidden_dim: int, size: int, d_state: int = 16, num_direction: int = 8,
                 chunk_size: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim, size=size, d_state=d_state,
                          num_direction=num_direction, chunk_size=chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ss2d(self.ln(x))


class LSSModule(nn.Module):

    def __init__(self, hidden_dim: int, size: int, depth: int = 3, d_state: int = 16,
                 num_direction: int = 8, use_checkpoint: bool = False, chunk_size: int = 256):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.hss_blocks = nn.ModuleList([
            HSSBlock(hidden_dim, size=size, d_state=d_state, num_direction=num_direction, chunk_size=chunk_size)
            for _ in range(depth)
        ])
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.permute(0, 2, 3, 1)
        for block in self.hss_blocks:
            if self.use_checkpoint and self.training:
                g = checkpoint.checkpoint(block, g, use_reentrant=False)
            else:
                g = block(g)
        g = g.permute(0, 3, 1, 2)

        l5 = self.conv5(x)
        l7 = self.conv7(x)

        fused = self.fuse(torch.cat([g, l5, l7], dim=1))
        return fused + x


class LSSStage(nn.Module):

    def __init__(self, hidden_dim: int, size: int, total_depth: int, d_state: int = 16,
                 num_direction: int = 8, use_checkpoint: bool = False, chunk_size: int = 256):
        super().__init__()
        if total_depth % 3 == 0:
            inner_depth, n_modules = 3, total_depth // 3
        elif total_depth % 2 == 0:
            inner_depth, n_modules = 2, total_depth // 2
        else:
            raise ValueError(f"decoder depth {total_depth} must be divisible by 2 or 3")
        self.modules_ = nn.ModuleList([
            LSSModule(hidden_dim, size=size, depth=inner_depth, d_state=d_state, num_direction=num_direction,
                      use_checkpoint=use_checkpoint, chunk_size=chunk_size)
            for _ in range(n_modules)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.modules_:
            x = m(x)
        return x


class MambaDecoder(nn.Module):

    def __init__(self, base_channels: int = 64, bottleneck_size: int = 8,
                 depths: List[int] = (3, 4, 6, 3), d_state: int = 16, num_direction: int = 8,
                 use_checkpoint: bool = False, chunk_size: int = 256):
        super().__init__()
        assert len(depths) == 4, "decoder depths must have 4 entries: [bottleneck, scale3, scale2, scale1]"
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        s3, s2, s1 = bottleneck_size * 2, bottleneck_size * 4, bottleneck_size * 8

        self.bottleneck_stage = LSSStage(c4, size=bottleneck_size, total_depth=depths[0], d_state=d_state,
                                          num_direction=num_direction, use_checkpoint=use_checkpoint, chunk_size=chunk_size)

        self.upsample1 = self._make_upsample(c4, c3)
        self.stage1 = LSSStage(c3, size=s3, total_depth=depths[1], d_state=d_state, num_direction=num_direction,
                                use_checkpoint=use_checkpoint, chunk_size=chunk_size)

        self.upsample2 = self._make_upsample(c3, c2)
        self.stage2 = LSSStage(c2, size=s2, total_depth=depths[2], d_state=d_state, num_direction=num_direction,
                                use_checkpoint=use_checkpoint, chunk_size=chunk_size)

        self.upsample3 = self._make_upsample(c2, c1)
        self.stage3 = LSSStage(c1, size=s1, total_depth=depths[3], d_state=d_state, num_direction=num_direction,
                                use_checkpoint=use_checkpoint, chunk_size=chunk_size)

    @staticmethod
    def _make_upsample(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.bottleneck_stage(x)

        scale3 = self.stage1(self.upsample1(x))
        scale2 = self.stage2(self.upsample2(scale3))
        scale1 = self.stage3(self.upsample3(scale2))
        return [scale1, scale2, scale3]
