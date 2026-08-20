import argparse
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import yaml

from src.data.datamodule import CrackDataModule
from src.models.factory import build_model, model_class_for

from src.explainability.grad_cam import GradCAM, upsample_cam_to_image
from src.explainability.shap_utils import (
    make_predict_fn_from_torch,
    shap_explain_resnet_superpixels_kernel,
    shap_explain_superpixels,
    shap_explain_vit_patches_kernel,
)
from src.explainability.utils import (
    denormalize_imagenet,
    to_uint8_hwc,
    colorize_heatmap,
    overlay_heatmap_on_image,
    save_png,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_from_checkpoint_if_any(cfg: Dict[str, Any], model):
    ckpt_path = cfg.get("checkpoint_path", None)
    if not ckpt_path:
        return model
    model_cls = model_class_for(cfg.get("name", ""))
    return model_cls.load_from_checkpoint(str(ckpt_path), config=cfg)


def pick_gradcam_target_layer(model_name: str, timm_model: torch.nn.Module) -> torch.nn.Module:
    """
    Select a reasonable default target layer for timm models.
    """
    model_name = model_name.lower()

    if "vit" in model_name:
        # token stream hook; outputs (B,N,D)
        return timm_model.blocks[-1].norm1

    # timm resnet usually exposes .layer4
    if hasattr(timm_model, "layer4"):
        return timm_model.layer4[-1]

    # fallback: last child module
    children = list(timm_model.children())
    if not children:
        raise ValueError("Cannot find a target layer automatically.")
    return children[-1]


def vit_grid_from_name(model_name: str) -> Tuple[int, int]:
    """Patch-grid size for a timm ViT name, e.g. vit_base_patch16_224 -> (14, 14).

    Previously this ignored its argument and returned (14, 14), which is right
    only for patch16 at 224. Any other variant reshaped the token sequence into
    the wrong grid and produced a silently meaningless heatmap.
    """
    match = re.search(r"patch(\d+)_(\d+)", model_name)
    if match is None:
        raise ValueError(
            f"Cannot infer a patch grid from {model_name!r}. "
            "Expected a timm name like 'vit_base_patch16_224'."
        )
    patch, image_size = int(match.group(1)), int(match.group(2))
    if patch <= 0 or image_size % patch != 0:
        raise ValueError(f"Image size {image_size} is not divisible by patch size {patch}.")
    side = image_size // patch
    return (side, side)


def main(config_path: str):
    cfg = load_config(config_path)

    output_dir = Path(cfg.get("output_dir", "reports/explainability"))
    out_dir = output_dir / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Explainability settings
    n_images = int(cfg.get("n_explain_images", 12))
    alpha = float(cfg.get("overlay_alpha", 0.45))

    do_shap = bool(cfg.get("do_shap", True))
    do_shap_vit_patch = bool(cfg.get("do_shap_vit_patch", False))

    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] n_explain_images={n_images}, overlay_alpha={alpha}")
    print(f"[INFO] do_shap={do_shap}, do_shap_vit_patch={do_shap_vit_patch}")

    # Data
    print("[INFO] Building datamodule...")

    # Data
    datamodule = CrackDataModule(
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 4)),
        preprocessing=cfg.get("preprocessing"),
        manifest_path=cfg.get("manifest_path", "data/processed/manifests/manifest.csv"),
        train_split_path=cfg.get("train_split_path", "data/processed/splits/train.csv"),
        val_split_path=cfg.get("val_split_path", "data/processed/splits/val.csv"),
        test_split_path=cfg.get("test_split_path", "data/processed/splits/test.csv"),
        raw_root=cfg.get("raw_root", "."),
    )

    print("[INFO] Setting up datamodule (stage=test)...")
    datamodule.setup(stage="test")
    dl = datamodule.test_dataloader()
    print("[INFO] Dataloader ready.")

    # Model
    print("[INFO] Building model...")
    model = build_model(cfg)
    model = load_from_checkpoint_if_any(cfg, model)
    
    model.eval().to(device)

    timm_model = model.model  # your LightningModules store timm model here
    model_name = str(cfg.get("model", "resnet50"))
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Checkpoint: {cfg.get('checkpoint_path', None)}")

    # Grad-CAM setup
    target_layer = pick_gradcam_target_layer(model_name, timm_model)
    print(f"[INFO] Grad-CAM target layer: {target_layer.__class__.__name__}")
    gc = GradCAM(timm_model, target_layer=target_layer)

    # SHAP setup
    # For superpixel SHAP we need uint8 RGB; predict_fn handles normalization internally
    predict_fn = make_predict_fn_from_torch(timm_model, device)

    do_shap = bool(cfg.get("do_shap", True))
    do_shap_vit_patch = bool(cfg.get("do_shap_vit_patch", False))
    shap_max_evals = int(cfg.get("shap_max_evals", 500))
    shap_batch_size = int(cfg.get("shap_batch_size", 16))

    if do_shap:
        print(f"[INFO] SHAP settings: shap_max_evals={shap_max_evals}, shap_batch_size={shap_batch_size}")
        if "vit" in model_name.lower():
            print(
                "[INFO] ViT SHAP settings: "
                f"vit_patch_size={int(cfg.get('vit_patch_size', 16))}, "
                f"vit_patch_shap_nsamples={int(cfg.get('vit_patch_shap_nsamples', 300))}, "
                f"vit_patch_shap_baseline={str(cfg.get('vit_patch_shap_baseline', 'blur'))}"
            )
        else:
            print(
                "[INFO] ResNet SHAP settings: "
                f"shap_n_segments={int(cfg.get('shap_n_segments', 80))}, "
                f"shap_compactness={float(cfg.get('shap_compactness', 10.0))}, "
                f"shap_nsamples={int(cfg.get('shap_nsamples', 500))}, "
                f"shap_baseline={str(cfg.get('shap_baseline', 'blur'))}"
            )

    # Iterate over test data, save explanations
    print("[INFO] Starting explanation loop...")

    # Iterate over test data, save explanations
    saved = 0
    for batch_idx, batch in enumerate(dl):
        x, y = batch
        x = x.to(device)

        B = x.shape[0]
        for i in range(B):
            if saved >= n_images:
                break

            xi = x[i : i + 1]  # (1,3,224,224)
            yi = int(y[i])

            print(f"[INFO] [{saved+1}/{n_images}] Explaining sample batch={batch_idx} idx={i} label={yi}...")

            # Convert to displayable RGB
            img01 = denormalize_imagenet(xi[0]).detach()
            img_u8 = to_uint8_hwc(img01)  # (224,224,3)

            # ---------- Grad-CAM ----------
            if "vit" in model_name.lower():
                cam_out = gc(xi, vit_grid=vit_grid_from_name(model_name))
            else:
                cam_out = gc(xi)

            cam224 = upsample_cam_to_image(cam_out.cam, (224, 224))[0, 0].detach().cpu().numpy()  # (224,224)
            hm_u8 = colorize_heatmap(cam224)
            overlay = overlay_heatmap_on_image(img_u8, hm_u8, alpha=alpha)

            stem = f"sample_{saved:04d}_label{yi}_pred{int(cam_out.class_idx[0].item())}"
            save_png(out_dir / f"{stem}_image.png", img_u8)
            save_png(out_dir / f"{stem}_gradcam_overlay.png", overlay)
            save_png(out_dir / f"{stem}_gradcam_heatmap.png", hm_u8)
            print(f"[INFO] [{saved+1}/{n_images}] Grad-CAM saved ({stem}).")

            # ---------- SHAP (practical) ----------
            if do_shap:
                print(f"[INFO] [{saved+1}/{n_images}] SHAP starting (may be slow)...")
                try:
                    if "vit" in model_name.lower():
                        # For ViT, use patch SHAP (token-aligned). Usually explain very few images.
                        patch_out = shap_explain_vit_patches_kernel(
                            timm_model,
                            device,
                            xi[0].detach(),
                            patch_size=int(cfg.get("vit_patch_size", 16)),
                            vit_grid=vit_grid_from_name(model_name),
                            nsamples=int(cfg.get("vit_patch_shap_nsamples", 300)),
                            baseline=str(cfg.get("vit_patch_shap_baseline", "blur")),
                        )
                        patch_map = np.abs(patch_out.values[0])  # (14,14)
                        patch_map = patch_map - patch_map.min()
                        patch_map = patch_map / (patch_map.max() + 1e-6)

                        patch_map_t = torch.from_numpy(patch_map).float()[None, None, :, :]
                        patch_map_224 = torch.nn.functional.interpolate(
                            patch_map_t, size=(224, 224), mode="bilinear", align_corners=False
                        )[0, 0].numpy()

                        shap_hm_u8 = colorize_heatmap(patch_map_224)
                    else:
                        # For ResNet, use superpixel KernelSHAP (feasible)
                        sp_out = shap_explain_resnet_superpixels_kernel(
                            timm_model,
                            device,
                            img_u8,
                            n_segments=int(cfg.get("shap_n_segments", 80)),
                            compactness=float(cfg.get("shap_compactness", 10.0)),
                            nsamples=int(cfg.get("shap_nsamples", 500)),
                            baseline=str(cfg.get("shap_baseline", "blur")),
                        )
                        shap_map = np.abs(sp_out.values[0])  # (224,224)
                        shap_map = shap_map - shap_map.min()
                        shap_map = shap_map / (shap_map.max() + 1e-6)
                        shap_hm_u8 = colorize_heatmap(shap_map)

                    shap_overlay = overlay_heatmap_on_image(img_u8, shap_hm_u8, alpha=alpha)
                    save_png(out_dir / f"{stem}_shap_overlay.png", shap_overlay)
                    save_png(out_dir / f"{stem}_shap_heatmap.png", shap_hm_u8)
                    print(f"[INFO] [{saved+1}/{n_images}] SHAP saved ({stem}).")

                except Exception as e:
                    print(f"[WARN] [{saved+1}/{n_images}] SHAP failed for {stem}: {type(e).__name__}: {e}")

            # ---------- SHAP (ViT patch KernelSHAP) ----------
            # WARNING: slow. Default is off.
            if do_shap_vit_patch and "vit" in model_name.lower():
                print(f"[INFO] [{saved+1}/{n_images}] Extra ViT patch SHAP starting (very slow)...")
                patch_out = shap_explain_vit_patches_kernel(
                    timm_model,
                    device,
                    xi[0].detach(),  # (3,224,224)
                    patch_size=int(cfg.get("vit_patch_size", 16)),
                    vit_grid=vit_grid_from_name(model_name),
                    nsamples=int(cfg.get("vit_patch_shap_nsamples", 300)),
                    baseline=str(cfg.get("vit_patch_shap_baseline", "blur")),
                )
                patch_map = np.abs(patch_out.values[0])  # (14,14)
                patch_map = patch_map - patch_map.min()
                patch_map = patch_map / (patch_map.max() + 1e-6)

                # upsample to 224x224 for overlay
                patch_map_t = torch.from_numpy(patch_map).float()[None, None, :, :]
                patch_map_224 = torch.nn.functional.interpolate(
                    patch_map_t, size=(224, 224), mode="bilinear", align_corners=False
                )[0, 0].numpy()

                patch_hm_u8 = colorize_heatmap(patch_map_224)
                patch_overlay = overlay_heatmap_on_image(img_u8, patch_hm_u8, alpha=alpha)
                save_png(out_dir / f"{stem}_shap_vitpatch_overlay.png", patch_overlay)
                save_png(out_dir / f"{stem}_shap_vitpatch_heatmap.png", patch_hm_u8)
                print(f"[INFO] [{saved+1}/{n_images}] Extra ViT patch SHAP saved ({stem}).")

            saved += 1

        if saved >= n_images:
            break

    gc.close()
    print(f"Saved {saved} explanation sets to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Generate Grad-CAM and SHAP visual explanations.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (reuse eval config).")
    args = p.parse_args()
    main(args.config)