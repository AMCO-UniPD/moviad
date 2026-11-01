"""
Weight loading utilities for DiAD in MoViAD.
Code adapted from:
    Title: DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection
    Authors: Haoyang He et al.
    URL: https://github.com/lewandofskee/DiAD (build_model.py)
    License: Apache-2.0
"""

import pickle
import types
from pathlib import Path
from typing import Dict, Optional

import torch
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
from transformers import CLIPTextConfig, CLIPTextModel

from moviad.models.diad.diad import DiAD

CLIP_PREFIX = "cond_stage_model.transformer."
EMPTY_PROMPT_IDS = [49406] + [49407] * 76


class _Ignored:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        return _Ignored


_restricted_pickle = types.ModuleType("restricted_pickle")
_restricted_pickle.Unpickler = _RestrictedUnpickler
_restricted_pickle.UnpicklingError = pickle.UnpicklingError
_restricted_pickle.load = pickle.load


def read_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        print(f"{path} contains non-tensor objects, loading tensors only with a restricted unpickler")
        state = torch.load(path, map_location="cpu", pickle_module=_restricted_pickle, weights_only=False)
    return state.get("state_dict", state)


def empty_prompt_embedding(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
    )
    text_model = CLIPTextModel(config).eval()
    model_keys = set(text_model.state_dict())
    clip_state = {}
    for key, value in state_dict.items():
        if not key.startswith(CLIP_PREFIX):
            continue
        name = key[len(CLIP_PREFIX):]
        for candidate in (name, name.removeprefix("text_model."), "text_model." + name):
            if candidate in model_keys:
                clip_state[candidate] = value
                break
    missing, unexpected = text_model.load_state_dict(clip_state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise KeyError(f"CLIP weights do not match: missing={missing[:5]}, unexpected={unexpected[:5]}")
    with torch.no_grad():
        return text_model(input_ids=torch.tensor([EMPTY_PROMPT_IDS])).last_hidden_state


def load_stable_diffusion(
    model: DiAD,
    checkpoint_path: str,
    autoencoder_path: Optional[str] = None,
) -> DiAD:
    state = read_state_dict(checkpoint_path)

    model.unet.load_state_dict(convert_ldm_unet_checkpoint(state, dict(model.unet.config)))
    model.init_sg_from_unet()

    autoencoder_state = read_state_dict(autoencoder_path) if autoencoder_path else state
    model.vae.load_state_dict(convert_ldm_vae_checkpoint(autoencoder_state, dict(model.vae.config)))

    model.set_text_embedding(empty_prompt_embedding(state))
    return model
