"""DINOv3 global image features for visual place seeding."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

DINOV3_REPO = Path(os.environ.get("MODELS_DIR", "") + "/source/dinov3")
DINOV3_WEIGHTS = Path(
    os.environ.get("MODELS_DIR", "") + "/facebook/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
INPUT_SIZE = 224
_MODEL = None
_DEVICE = None


def available() -> bool:
    return DINOV3_REPO.joinpath("hubconf.py").is_file() and DINOV3_WEIGHTS.is_file()


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    import torch

    if not available():
        raise FileNotFoundError(f"DINOv3 weights missing: {DINOV3_WEIGHTS}")
    _DEVICE = _device()
    model = torch.hub.load(
        repo_or_dir=str(DINOV3_REPO),
        model="dinov3_vitl16",
        source="local",
        pretrained=False,
    )
    try:
        state = torch.load(DINOV3_WEIGHTS, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(DINOV3_WEIGHTS, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(_DEVICE).eval()
    _MODEL = model
    return model


def _preprocess(rgb: np.ndarray):
    import torch

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1)
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def encode_rgbs(rgbs: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if not rgbs:
        return np.zeros((0, 1), dtype=np.float32)
    model = load_model()
    chunks = []
    for start in range(0, len(rgbs), batch_size):
        batch = torch.stack([_preprocess(rgb) for rgb in rgbs[start : start + batch_size]])
        batch = batch.to(_DEVICE)
        with torch.no_grad():
            outputs = model.forward_features(batch)
        if isinstance(outputs, dict):
            features = outputs.get("x_norm_clstoken")
            if features is None:
                features = outputs.get("x_norm_patchtokens")
                if features is not None:
                    features = features.mean(dim=1)
        else:
            features = outputs
        if features.ndim > 2:
            features = F.adaptive_avg_pool1d(features.flatten(1).unsqueeze(1), 1024).squeeze(1)
        features = F.normalize(features.float(), dim=1)
        chunks.append(features.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def encode_rgb(rgb: np.ndarray) -> np.ndarray:
    return encode_rgbs([rgb])[0]
