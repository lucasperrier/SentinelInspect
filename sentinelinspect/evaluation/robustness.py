import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

EPS = 1e-8
SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy'}


def _load_map(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.npy':
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path).convert('L'), dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + EPS)
    return arr


def _binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    target_bin = target.astype(bool)
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    return float(intersection / union) if union > 0 else 0.0


def localization_accuracy_from_dirs(cam_dir: Optional[str], mask_dir: Optional[str], threshold: float = 0.5) -> float:
    if not cam_dir or not mask_dir:
        return float('nan')

    cam_root = Path(cam_dir)
    mask_root = Path(mask_dir)
    if not cam_root.exists() or not mask_root.exists():
        return float('nan')

    scores = []
    for cam_path in cam_root.iterdir():
        if cam_path.suffix.lower() not in SUPPORTED_EXT:
            continue
        mask_path = mask_root / cam_path.name
        if not mask_path.exists():
            continue
        cam = _load_map(cam_path)
        mask = _load_map(mask_path)
        cam_bin = (cam >= threshold).astype(np.uint8)
        mask_bin = (mask >= threshold).astype(np.uint8)
        scores.append(_binary_iou(cam_bin, mask_bin))

    return float(np.mean(scores)) if scores else float('nan')


def faithfulness_drop_from_csv(csv_path: Optional[str]) -> float:
    if not csv_path:
        return float('nan')
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return float('nan')

    df = pd.read_csv(csv_file)
    required_cols = {'original_prob', 'perturbed_prob'}
    if not required_cols.issubset(df.columns):
        return float('nan')

    original = df['original_prob'].to_numpy(dtype=np.float32)
    perturbed = df['perturbed_prob'].to_numpy(dtype=np.float32)
    denom = np.maximum(np.abs(original), EPS)
    drops = np.clip(original - perturbed, a_min=0.0, a_max=None)
    score = drops / denom
    if math.isnan(score.mean()):
        return float('nan')
    return float(score.mean())