import timm
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
#from torch.optim import StableAdamW, WarmCosineScheduler
from torch.nn import functional as F

from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.models.dinomaly.components import bMlp, Block as VitBlock, LinearAttention2, global_cosine_hm_percent
from moviad.models.dinomaly.optimizers import StableAdamW
from moviad.models.dinomaly.scheduler import WarmCosineScheduler

import math
class DinomalyTrainArgs(TrainingArgs):

    total_iters: int = 5000
    lr_scheduler = None

    def init_train(self, model):
        if not self.optimizer:
            self.optimizer = StableAdamW(
                [{'params': model.trainable.parameters()}],
                lr=2e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
                amsgrad=True,
                eps=1e-8
            )

        if not self.lr_scheduler:
            self.lr_scheduler = WarmCosineScheduler(
                self.optimizer,
                base_value=2e-3,
                final_value=2e-4,
                total_iters=self.total_iters,
                warmup_iters=100
            )

        if not self.loss_function:
            self.loss_function = global_cosine_hm_percent

    def init_task(self, model, iters_per_epoch: int):
        # restart warmup + cosine schedule on the length of the current task
        total_iters = self.epochs * iters_per_epoch
        self.lr_scheduler = WarmCosineScheduler(
            self.optimizer,
            base_value=2e-3,
            final_value=2e-4,
            total_iters=total_iters,
            warmup_iters=min(100, total_iters)
        )

    def __to_dict__(self):

        return {
            "total_iters": self.total_iters,
        }

class Dinomaly(VADModel):
    def __init__(
        self,
        encoder_name: str,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
        mask_neighbor_size=0,
        remove_class_token=False,
    ) -> None:
        super(Dinomaly, self).__init__()

        self.encoder_name = encoder_name

        encoder = timm.create_model(self.encoder_name, pretrained=True, img_size=(224, 224))

        def prepare_tokens(x):
            x = encoder.patch_embed(x)
            x = encoder._pos_embed(x)
            if hasattr(encoder, 'norm_pre') and encoder.norm_pre is not None:
                x = encoder.norm_pre(x)
            return x

        encoder.prepare_tokens = prepare_tokens

        embed_dim = encoder.num_features
        num_heads = encoder.blocks[0].attn.num_heads

        bottleneck = []
        decoder = []

        bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
        bottleneck = nn.ModuleList(bottleneck)

        for i in range(8):
            blk = VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn_drop=0.,
                attn=LinearAttention2
            )
            decoder.append(blk)
        decoder = nn.ModuleList(decoder)

        self.trainable = nn.ModuleList([bottleneck, decoder])

        for m in self.trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.train_steps = 0

        self.mask_neighbor_size = mask_neighbor_size

    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        self.encoder.to(device)
        self.bottleneck.to(device)
        self.decoder.to(device)
        return self

    def train(self, mode = True):
        super().train(mode)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.bottleneck.train(mode)
        self.decoder.train(mode)
        return self

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder.prepare_tokens(x)
        en_list = []

        # encoder forward
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                x = blk(x)

            if i in self.target_layers:
                en_list.append(x)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_reg_tokens))

        x = self.fuse_feature(en_list)

        # bottleneck forward
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        # decoder forward
        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        # remove class token and reg tokens
        en = [e[:, 1 + self.encoder.num_reg_tokens:, :] for e in en]
        de = [d[:, 1 + self.encoder.num_reg_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        if self.training:
            return en, de
        else:
            out_size = (H, W)

            a_map_list = []
            for i in range(len(en)):
                a_map = 1 - F.cosine_similarity(en[i], de[i])
                a_map = torch.unsqueeze(a_map, dim=1)
                a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
                a_map_list.append(a_map)

            anomaly_maps = torch.cat(a_map_list, dim=1).mean(dim=1)
            return anomaly_maps, anomaly_maps.amax(dim=(1,2))

    def compute_loss(self, batch: torch.Tensor, training_args: DinomalyTrainArgs) -> torch.Tensor:
        batch = batch.to(self.device)
        en, de = self.forward(batch)

        p_final = 0.9
        p = min(p_final * self.train_steps / 1000, p_final)
        return training_args.loss_function(en, de, p=p, factor=0.1)

    def train_step(self, batch: torch.Tensor, training_args: DinomalyTrainArgs):
        loss = self.compute_loss(batch, training_args)

        training_args.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=0.1)
        training_args.optimizer.step()
        training_args.lr_scheduler.step()
        self.train_steps += 1

        return loss.item()

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        bytes_per_param = 4
        total_size_mb = (total_params * bytes_per_param) / (1024 ** 2)

        return total_size_mb
