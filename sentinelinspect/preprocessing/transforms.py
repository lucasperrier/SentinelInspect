from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULTS: dict[str, Any] = {
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "hflip_p": 0.5,
    "brightness_contrast_p": 0.3,
    "shift_scale_rotate_p": 0.3,
    "shift_limit": 0.03,
    "scale_limit": 0.05,
    "rotate_limit": 10,
}


def resolve_preprocessing(pre: dict[str, Any] | None) -> dict[str, Any]:
    """Fill in defaults for a preprocessing block.

    Takes the preprocessing settings themselves, not the config tree that
    contains them. The previous signature expected a dict with a
    ``preprocessing`` key, so every caller that passed the block directly
    silently got defaults instead.
    """
    pre = pre or {}
    unknown = set(pre) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown preprocessing keys: {sorted(unknown)}. "
            f"Expected a subset of {sorted(DEFAULTS)}"
        )

    resolved = {**DEFAULTS, **pre}
    resolved["image_size"] = int(resolved["image_size"])
    resolved["mean"] = list(resolved["mean"])
    resolved["std"] = list(resolved["std"])
    for key in ("hflip_p", "brightness_contrast_p", "shift_scale_rotate_p", "shift_limit", "scale_limit"):
        resolved[key] = float(resolved[key])
    resolved["rotate_limit"] = int(resolved["rotate_limit"])
    return resolved


def build_train_transforms(pre: dict[str, Any] | None = None) -> A.Compose:
    p = resolve_preprocessing(pre)
    return A.Compose([
        A.Resize(p["image_size"], p["image_size"]),
        A.HorizontalFlip(p=p["hflip_p"]),
        A.RandomBrightnessContrast(p=p["brightness_contrast_p"]),
        A.Affine(
            translate_percent=(-p["shift_limit"], p["shift_limit"]),
            scale=(1.0 - p["scale_limit"], 1.0 + p["scale_limit"]),
            rotate=(-p["rotate_limit"], p["rotate_limit"]),
            p=p["shift_scale_rotate_p"],
        ),
        A.Normalize(mean=p["mean"], std=p["std"]),
        ToTensorV2(),
    ])


def build_val_transforms(pre: dict[str, Any] | None = None) -> A.Compose:
    """Deterministic pipeline: resize, normalise, to tensor. No augmentation.

    Evaluation and inference must both use this, or the model sees a different
    distribution at serving time than it was scored on.
    """
    p = resolve_preprocessing(pre)
    return A.Compose([
        A.Resize(p["image_size"], p["image_size"]),
        A.Normalize(mean=p["mean"], std=p["std"]),
        ToTensorV2(),
    ])


def build_eval_transforms(pre: dict[str, Any] | None = None) -> A.Compose:
    return build_val_transforms(pre)


def build_inference_transforms(pre: dict[str, Any] | None = None) -> A.Compose:
    return build_val_transforms(pre)
