"""
UniAD Transformer
Code adapted from:
    Title: Towards Unsupervised Anomaly Detection (UniAD)
    Authors: Zhiyuan You et al.
    URL: https://github.com/zhiyuanyou/UniAD
    License: MIT
"""

import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



# Weight Initialization
def initialize_from_cfg(model, cfg):
    if cfg is None:
        _initialize(model, "normal", std=0.01)
        return
    cfg = copy.deepcopy(cfg)
    method = cfg.pop("method")
    _initialize(model, method, **cfg)


def _initialize(model, method, **kwargs):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    if method == "normal":
        _init_weights_normal(model, **kwargs)
    elif "xavier" in method:
        _init_weights_xavier(model, method)
    elif "msra" in method:
        _init_weights_msra(model, method)
    else:
        raise NotImplementedError(f"{method} not supported")


def _init_weights_normal(module, std=0.01):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def _init_weights_xavier(module, method):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            if "uniform" in method:
                nn.init.xavier_uniform_(m.weight.data)
            else:
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def _init_weights_msra(module, method):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            if "normal" in method:
                nn.init.kaiming_normal_(m.weight.data, a=1)
            else:
                nn.init.kaiming_uniform_(m.weight.data, a=1)
            if m.bias is not None:
                m.bias.data.zero_()


# Position Embedding
class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding for 2D feature maps."""
    def __init__(self, feature_size, num_pos_feats=128, temperature=10000,
                 normalize=False, scale=None):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)
        return pos.to(tensor.device)


class PositionEmbeddingLearned(nn.Module):
    """Learned absolute position embedding."""
    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)
        j = torch.arange(self.feature_size[0], device=tensor.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            torch.cat([x_emb.unsqueeze(0)] * self.feature_size[0], dim=0),
            torch.cat([y_emb.unsqueeze(1)] * self.feature_size[1], dim=1),
        ], dim=-1).flatten(0, 1)
        return pos


def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        return PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        return PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    raise ValueError(f"not supported {pos_embed_type}")



# Transformer Components
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu": return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self attention and feedforward network."""
    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with learned query embeddings."""
    def __init__(self, hidden_dim, feature_size, nhead, dim_feedforward,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, out, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                          src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None):
        output = memory
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                          memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask,
                          pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    """
    Full transformer with encoder and decoder.Includes neighbor mask for local attention.
    """

    def __init__(self, hidden_dim, feature_size, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout=0.1,
                 activation="relu", normalize_before=False,
                 neighbor_mask=None):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, feature_size, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # feature_size/neighbor_size never change after construction
        if neighbor_mask is not None:
            self.register_buffer(
                "_neighbor_attn_mask",
                self.generate_mask(feature_size, neighbor_mask["neighbor_size"], device="cpu"),
            )

    def generate_mask(self, feature_size, neighbor_size, device):
        """Generate attention mask to restrict each token to attend only to its neighbors."""
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end] = 0
        mask = mask.view(h * w, h * w)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src, pos_embed):
        _, batch_size, _ = src.shape
        pos_embed = torch.cat([pos_embed.unsqueeze(1)] * batch_size, dim=1)

        if self.neighbor_mask is not None:
            mask = self._neighbor_attn_mask
            mask_enc = mask if self.neighbor_mask["mask"][0] else None
            mask_dec1 = mask if self.neighbor_mask["mask"][1] else None
            mask_dec2 = mask if self.neighbor_mask["mask"][2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder(src, mask=mask_enc, pos=pos_embed)
        output_decoder = self.decoder(
            output_encoder,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed
        )
        return output_decoder, output_encoder


class UniADCore(nn.Module):
    """UniAD's core transformer-based reconstruction module."""
    def __init__(self, inplanes, instrides, feature_size, hidden_dim,
                 pos_embed_type, initializer, neighbor_mask=None,
                 feature_jitter=None, **kwargs):
        super().__init__()
        self.feature_size = feature_size
        self.feature_jitter = feature_jitter
        self.pos_embed = build_position_embedding(pos_embed_type, feature_size, hidden_dim)
        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask=neighbor_mask, **kwargs
        )
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])
        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        """Add random noise to feature tokens during training for regularization."""
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            jitter = torch.randn_like(feature_tokens)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, feature_align):
        feature_tokens = rearrange(feature_align, "b c h w -> (h w) b c")
        if self.training and self.feature_jitter is not None:
            feature_tokens = self.add_jitter(
                feature_tokens,
                self.feature_jitter["scale"],
                self.feature_jitter["prob"]
            )
        feature_tokens = self.input_proj(feature_tokens)
        pos_embed = self.pos_embed(feature_tokens)
        output_decoder, _ = self.transformer(feature_tokens, pos_embed)
        feature_rec_tokens = self.output_proj(output_decoder)
        feature_rec = rearrange(feature_rec_tokens, "(h w) b c -> b c h w",
                                h=self.feature_size[0])
        pred = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2,
                                    dim=1, keepdim=True))
        pred = self.upsample(pred)
        return feature_rec, pred


# MFCN Neck
class MFCN(nn.Module):

    def __init__(self, inplanes, instrides, outstrides):
        super().__init__()
        assert isinstance(inplanes, list)
        assert isinstance(outstrides, list) and len(outstrides) == 1

        self.inplanes = inplanes
        self.outplanes = [sum(inplanes)]
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [
            in_stride / outstrides[0] for in_stride in instrides
        ]
        self.upsample_list = nn.ModuleList([
            nn.UpsamplingBilinear2d(scale_factor=sf)
            for sf in self.scale_factors
        ])

    def forward(self, features):
        assert len(self.inplanes) == len(features)
        feature_list = []
        for i, feat in enumerate(features):
            feature_list.append(self.upsample_list[i](feat))
        return torch.cat(feature_list, dim=1)

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides
