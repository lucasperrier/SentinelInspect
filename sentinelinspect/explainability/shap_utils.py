from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SHAPOutput:
    """
    values: SHAP attributions.
      - superpixel SHAP: typically (B, H, W, 3) or (B, H, W, C)
      - patch SHAP: (B, H_patches, W_patches)
    base_values: expected value(s) from SHAP
    target_class: explained classes
    """
    values: np.ndarray
    base_values: np.ndarray
    target_class: int | np.ndarray


def make_predict_fn_from_torch(
    model: torch.nn.Module,
    device: torch.device,
    *,
    assumes_uint8_rgb: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns predict(images_np) -> probs_np for SHAP.

    SHAP image maskers typically feed uint8 RGB images (B,H,W,3).
    Your training expects normalized tensors:
      mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] and NCHW.

    This function converts uint8->float, normalizes, and runs model.
    """
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def predict(images_np: np.ndarray) -> np.ndarray:
        if images_np.ndim == 3:
            images_np_b = images_np[None, ...]
        else:
            images_np_b = images_np

        x = torch.from_numpy(images_np_b).to(device)

        # (B,H,W,3) -> (B,3,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()

        if assumes_uint8_rgb:
            x = x.float() / 255.0
        else:
            x = x.float()

        x = (x - mean) / std

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    return predict


def _unpack_kernel_shap(shap_vals, expected_value, n_features: int):
    """Normalise KernelExplainer output across SHAP versions.

    shap < 0.45 returned (1, k) for a single-output model, or a list of arrays
    for multi-output. Newer versions return (1, k, n_outputs) even when there
    is one output, so the per-feature entry became a 1-element array. NumPy 2
    no longer converts those to scalars implicitly, which surfaced as
    "only 0-dimensional arrays can be converted to Python scalars".
    """
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    values = np.asarray(shap_vals, dtype=np.float64)
    values = values.reshape(-1) if values.size == n_features else values.reshape(n_features, -1)[:, 0]

    base = np.asarray(expected_value, dtype=np.float64).reshape(-1)
    base_val = float(base[0]) if base.size else 0.0
    return values.astype(np.float32), base_val


def shap_explain_superpixels(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    images_uint8: np.ndarray,
    *,
    target_class: int | None = None,
    max_evals: int = 500,
    batch_size: int = 16,
    segmentation: str = "slic",
) -> SHAPOutput:
    """
    Superpixel SHAP using shap.maskers.Image.
    Realistic usage: small number of images (e.g., 1..20) because it's slow.
    """
    import shap

    if images_uint8.ndim == 3:
        images_uint8 = images_uint8[None, ...]

    masker = shap.maskers.Image(segmentation, images_uint8[0].shape)
    explainer = shap.PermutationExplainer(predict_fn, masker)

    if target_class is None:
        probs = predict_fn(images_uint8)
        target_arr = probs.argmax(axis=1)
    else:
        target_arr = np.full((images_uint8.shape[0],), int(target_class), dtype=np.int64)

    shap_values = explainer(
        images_uint8,
        max_evals=max_evals,
        batch_size=batch_size,
        outputs=target_arr.tolist(),
    )

    return SHAPOutput(
        values=np.array(shap_values.values),
        base_values=np.array(shap_values.base_values),
        target_class=target_arr,
    )

def shap_explain_resnet_superpixels_kernel(
    model: torch.nn.Module,
    device: torch.device,
    image_uint8: np.ndarray,
    *,
    target_class: int | None = None,
    n_segments: int = 80,
    compactness: float = 10.0,
    nsamples: int = 500,
    baseline: str = "blur",
) -> SHAPOutput:
    """
    Practical SHAP for CNNs: KernelSHAP over SLIC superpixels.
    Explains a *single* image (H,W,3) uint8.

    Returns values as a per-pixel map (H,W) by assigning each superpixel its SHAP value.
    """
    import shap
    from skimage.segmentation import slic  # requires scikit-image

    assert image_uint8.ndim == 3 and image_uint8.shape[2] == 3, "Expected (H,W,3) uint8 image."
    H, W, _ = image_uint8.shape

    # 1) superpixels
    segments = slic(image_uint8, n_segments=n_segments, compactness=compactness, start_label=0)
    k = int(segments.max() + 1)

    # 2) build baseline image (uint8)
    if baseline == "zero":
        base = np.zeros_like(image_uint8)
    else:
        # cheap blur baseline without extra deps: downsample/upsample
        small = torch.from_numpy(image_uint8).float().permute(2, 0, 1)[None, ...] / 255.0
        small = F.avg_pool2d(small, kernel_size=16, stride=16)
        base01 = F.interpolate(small, size=(H, W), mode="nearest")[0].permute(1, 2, 0).numpy()
        base = (base01 * 255.0).clip(0, 255).astype(np.uint8)

    # 3) mask -> image
    def apply_mask(mask01: np.ndarray) -> np.ndarray:
        mask01 = mask01.astype(np.float32)
        out = image_uint8.copy()
        # turn off superpixels
        off = np.where(mask01 < 0.5)[0]
        for sp in off:
            out[segments == sp] = base[segments == sp]
        return out

    # 4) prediction function in mask space
    predict_img = make_predict_fn_from_torch(model, device, assumes_uint8_rgb=True)

    # Determine target class
    if target_class is None:
        probs = predict_img(image_uint8[None, ...])
        tgt = int(probs.argmax(axis=1)[0])
    else:
        tgt = int(target_class)

    def f(masks: np.ndarray) -> np.ndarray:
        # masks: (N,k)
        imgs = np.stack([apply_mask(m) for m in masks], axis=0)
        probs = predict_img(imgs)  # (N,num_classes)
        return probs[:, [tgt]]     # SHAP expects 2D output

    # Background: a few random masks
    bg = np.random.binomial(1, 0.5, size=(min(20, k), k)).astype(np.float32)
    ke = shap.KernelExplainer(f, bg)

    shap_vals = ke.shap_values(np.ones((1, k), dtype=np.float32), nsamples=nsamples)
    vals, base_val = _unpack_kernel_shap(shap_vals, ke.expected_value, k)

    # Convert superpixel attributions -> per-pixel heatmap
    map_hw = np.zeros((H, W), dtype=np.float32)
    for sp in range(k):
        map_hw[segments == sp] = float(vals[sp])

    return SHAPOutput(
        values=map_hw[None, ...],  # (1,H,W)
        base_values=np.array([base_val]),
        target_class=tgt,
    )


def shap_explain_vit_patches_kernel(
    model: torch.nn.Module,
    device: torch.device,
    image_t: torch.Tensor,
    *,
    patch_size: int = 16,
    vit_grid: tuple[int, int] = (14, 14),
    target_class: int | None = None,
    nsamples: int = 300,
    baseline: str = "blur",
) -> SHAPOutput:
    """
    Patch-level KernelSHAP for a *single* image.

    image_t: (3,224,224) or (1,3,224,224) normalized like your datamodule output.
    Returns: values shaped (1, H_p, W_p).
    """
    import shap

    model.eval()

    if image_t.ndim == 3:
        image_t = image_t.unsqueeze(0)

    image_t = image_t.to(device)
    B, C, H, W = image_t.shape
    H_p, W_p = vit_grid
    M = H_p * W_p

    if H_p * patch_size != H or W_p * patch_size != W:
        raise ValueError("vit_grid and patch_size must match the image resolution (e.g., 224=14*16).")

    # baseline image in model space (already normalized!)
    with torch.no_grad():
        if baseline == "zero":
            base = torch.zeros_like(image_t)
        else:
            pooled = F.avg_pool2d(image_t, kernel_size=patch_size, stride=patch_size)
            base = F.interpolate(pooled, size=(H, W), mode="nearest")

    with torch.no_grad():
        logits = model(image_t)
        if target_class is None:
            tgt = int(logits.argmax(dim=1).item())
        else:
            tgt = int(target_class)

    def apply_mask(mask01: np.ndarray) -> torch.Tensor:
        # mask01: (M,) where 1 keeps original patch, 0 uses baseline patch
        m = torch.from_numpy(mask01.reshape(H_p, W_p)).float().to(device)
        m_up = m.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)  # (H,W)
        m_up = m_up.view(1, 1, H, W)
        return image_t * m_up + base * (1.0 - m_up)

    def predict_from_masks(masks: np.ndarray) -> np.ndarray:
        """
        KernelExplainer expects a 2D output. We explain a *single* scalar:
        p(class=tgt). So we must return shape (K, 1).
        """
        outs: list[float] = []
        with torch.no_grad():
            for mask in masks:
                x_masked = apply_mask(mask.astype(np.float32))
                probs = F.softmax(model(x_masked), dim=1)  # (1, num_classes)
                outs.append(float(probs[0, tgt].detach().cpu().item()))
        return np.asarray(outs, dtype=np.float32).reshape(-1, 1)

    # background masks: a few random masks for KernelExplainer baseline
    bg = np.random.binomial(1, 0.5, size=(min(20, M), M)).astype(np.float32)

    ke = shap.KernelExplainer(predict_from_masks, bg)

    # explain the "all-ones" mask (original image) in mask space
    shap_vals = ke.shap_values(np.ones((1, M), dtype=np.float32), nsamples=nsamples)

    vals_m, base_val = _unpack_kernel_shap(shap_vals, ke.expected_value, M)

    if vals_m.size != M:
        raise ValueError(f"Expected {M} patch attributions, got {vals_m.size}.")

    return SHAPOutput(
        values=vals_m.reshape(1, H_p, W_p),
        base_values=np.array([base_val], dtype=np.float32),
        target_class=tgt,
    )