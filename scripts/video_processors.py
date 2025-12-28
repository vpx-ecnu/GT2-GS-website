from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import numpy as np
from PIL import Image
import imageio
import torch

# ✅ 内置 processor 函数库（可自由增删）
PROCESSORS: Dict[str, Callable] = {}

def register_processor(name: str):
    def decorator(func: Callable) -> Callable:
        PROCESSORS[name] = func
        return func
    return decorator

# --- 🔹 to_tensor: [F,H,W,C] uint8 → [F,C,H,W] float32 [0,1] ---
@register_processor("to_tensor")
def to_tensor(
    frames,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
    **kwargs
):
    if isinstance(frames, list):
        frames = torch.stack([torch.as_tensor(f) for f in frames])
    elif isinstance(frames, np.ndarray):
        frames = torch.as_tensor(frames)
    elif not isinstance(frames, torch.Tensor):
        raise TypeError(f"Unsupported type {type(frames)}")

    if frames.ndim == 4 and frames.shape[-1] == 3:  # [F,H,W,C]
        frames = frames.permute(0, 3, 1, 2)  # → [F,C,H,W]
    elif frames.ndim == 3 and frames.shape[0] == 3:  # [C,H,W] (single image)
        frames = frames.unsqueeze(0)

    if normalize:
        frames = frames.float() / 255.0
    else:
        frames = frames.float()

    return frames.to(device=device, dtype=dtype)

@register_processor("to_pil")
def to_pil(
    frames,
    **kwargs
):
    from torchvision.transforms.functional import to_pil_image

    return [to_pil_image(f) for f in frames]

# --- 🔹 resize: 支持 width/height/scale ---
@register_processor("resize")
def resize(
    frames,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
    resample: int = Image.LANCZOS,
    **kwargs
):
    from torchvision.transforms.functional import to_pil_image
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    if not isinstance(frames, list):
        frames = [to_pil_image(f) for f in frames]

    out = []
    for f in frames:
        w, h = f.size
        if scale is not None:
            w, h = int(w * scale), int(h * scale)
        else:
            w = width or w
            h = height or h
        resized = f.resize((w, h), resample=resample)
        out.append(to_tensor(resized))
    return torch.stack(out)

# --- 🔹 temporal_sample: step=2 → 取 0,2,4,... ---
@register_processor("temporal_sample")
def temporal_sample(frames, step: int = 2, **kwargs):
    return frames[::step]

# --- 🔹 center_crop ---
@register_processor("center_crop")
def center_crop(frames, output_size: tuple, **kwargs):
    from torchvision.transforms.functional import center_crop
    if isinstance(frames, torch.Tensor) and frames.ndim == 4:
        return torch.stack([center_crop(f, output_size) for f in frames])
    else:
        frames = [torch.as_tensor(f) for f in frames]
        return torch.stack([center_crop(f, output_size) for f in frames])

# --- 🔹 normalize (per-channel, only if already tensor) ---
@register_processor("normalize")
def normalize(frames, mean, std, **kwargs):
    if not isinstance(frames, torch.Tensor):
        raise TypeError("normalize requires torch.Tensor input (use 'to_tensor' first)")
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return (frames - mean) / std

# --- 🔹 limit_frames ---
@register_processor("limit_frames")
def limit_frames(frames, max_frames: int = 100, **kwargs):
    return frames[:max_frames]