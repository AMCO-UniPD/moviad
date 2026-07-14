import hashlib
import os
import urllib
import warnings
from typing import Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np

from .build_model import build_model
from moviad.backbones.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from moviad.backbones.clip.model_load import _download, _transform
from torchvision.transforms import InterpolationMode

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["load", "get_similarity_map",  "compute_similarity", "download_prompt_learner"]
_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

_CHECKPOINTS = {
    "visa": {
        "url": "https://github.com/zqhang/AnomalyCLIP/raw/3911738c0867544f545a076ad78f3f11d9ecbfdf/checkpoints/9_12_4_multiscale_visa/epoch_15.pth",
        "hash": "415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4" 
    },
    "mvtec": {
        "url": "https://github.com/zqhang/AnomalyCLIP/raw/3911738c0867544f545a076ad78f3f11d9ecbfdf/checkpoints/9_12_4_multiscale/epoch_15.pth",
        "hash": "94ce202da3e6486a864b904fdfed5057de75846c5834e446fd1d2fe7f97acb44"
    }
}


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", design_details = None, jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    print("name", name)
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(name, state_dict or model.state_dict(), design_details).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


def compute_similarity(image_features, text_features, t=2):
    prob_1 = image_features[:, :1, :] @ text_features.t()
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity/0.07).softmax(-1), prob_1

def download_prompt_learner(name: str, cache_dir: str = None) -> str:
    """
    Downloads learned prompts for AnomalyCLIP.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/anomaly_clip")

    if name.lower() not in _CHECKPOINTS:
        if os.path.isfile(name):
            return name
        raise ValueError(f"Model {name} not found. Available: {list(_CHECKPOINTS.keys())}")

    model_info = _CHECKPOINTS[name.lower()]
    url = model_info["url"]
    expected_hash = model_info["hash"]
    
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(cache_dir, filename)

    # Check if file exists and verify hash
    if os.path.exists(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_hash:
            return download_target
        else:
            warnings.warn(f"{filename} exists but the hash is invalid. Re-downloading...")

    # Start download using urllib
    print(f"Downloading {name} prompt learner weights to {download_target}...")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)

    # Final validation
    actual_hash = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not match.")

    return download_target
