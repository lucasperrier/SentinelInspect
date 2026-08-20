from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: (3,H,W) or (B,3,H,W) normalized with ImageNet stats
    returns same shape in [0,1]
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    out = (x * std) + mean
    out = out.clamp(0.0, 1.0)

    return out[0] if squeeze else out


def to_uint8_hwc(x01: torch.Tensor) -> np.ndarray:
    """
    x01: (3,H,W) or (B,3,H,W) in [0,1]
    returns uint8 RGB:
      - (H,W,3) or (B,H,W,3)
    """
    if x01.ndim == 3:
        x01 = x01.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    x = (x01 * 255.0).round().clamp(0, 255).byte()      # (B,3,H,W)
    x = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B,H,W,3)

    return x[0] if squeeze else x


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def colorize_heatmap(cam01_hw: np.ndarray) -> np.ndarray:
    """
    cam01_hw: (H,W) float in [0,1]
    returns uint8 RGB (H,W,3)
    """
    cv2 = _try_import_cv2()
    cam_u8 = (cam01_hw * 255.0).clip(0, 255).astype(np.uint8)

    if cv2 is not None:
        hm_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        hm_rgb = hm_bgr[:, :, ::-1]
        return hm_rgb

    # fallback: simple grayscale to RGB
    return np.stack([cam_u8, cam_u8, cam_u8], axis=-1)


def overlay_heatmap_on_image(
    image_uint8: np.ndarray,
    heatmap_uint8: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    image_uint8: (H,W,3) RGB
    heatmap_uint8: (H,W,3) RGB
    returns uint8 RGB overlay
    """
    img = image_uint8.astype(np.float32)
    hm = heatmap_uint8.astype(np.float32)
    out = (1.0 - alpha) * img + alpha * hm
    return out.clip(0, 255).astype(np.uint8)


def save_png(path: Path, rgb_uint8: np.ndarray) -> None:
    """
    Saves RGB uint8 using PIL.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8).save(path)