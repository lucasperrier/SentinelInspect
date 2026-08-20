from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class CAMOutput:
    """
    cam: normalized CAM in [0,1]
      - CNN case: (B, 1, H, W) at target layer resolution
      - ViT case: (B, 1, H_patches, W_patches) at patch resolution
    logits: model logits (B, num_classes)
    class_idx: explained class indices (B,)
    """
    cam: torch.Tensor
    logits: torch.Tensor
    class_idx: torch.Tensor


class GradCAM:
    """
    Grad-CAM for:
      - CNN featuremaps: activations (B,C,H,W)
      - ViT tokens:     activations (B,N,D) -> patch CAM (B,1,H_p,W_p)

    This is model-agnostic as long as you can provide a target_layer to hook.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        """Remove hooks (important in notebooks and repeated runs)."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _forward_hook(self, module, inp, out):
        # Save activations of target layer
        self._acts = out

    def _backward_hook(self, module, grad_input, grad_output):
        # Save gradient w.r.t. the target layer output
        self._grads = grad_output[0]

    @staticmethod
    def _normalize(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Normalize per-sample to [0,1]
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + eps)
        return cam

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[Union[int, torch.Tensor]] = None,
        *,
        vit_grid: Optional[Tuple[int, int]] = None,
    ) -> CAMOutput:
        """
        Compute CAM for input x.

        Args:
          x: (B,3,H,W) normalized tensor (as your datamodule provides)
          class_idx:
            - None => explain predicted class for each sample
            - int  => explain this class for all samples
            - tensor (B,) => per-sample classes
          vit_grid: (H_patches,W_patches) for ViT token CAM (required for token outputs)

        Returns:
          CAMOutput(cam, logits, class_idx)
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        if logits.ndim != 2:
            raise ValueError(f"Expected logits of shape (B,num_classes), got {tuple(logits.shape)}")

        B = logits.shape[0]

        if class_idx is None:
            class_idx_t = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx_t = torch.full((B,), int(class_idx), device=logits.device, dtype=torch.long)
        else:
            class_idx_t = class_idx.to(logits.device).long()
            if class_idx_t.numel() != B:
                raise ValueError(f"class_idx tensor must have B elements ({B}), got {class_idx_t.numel()}")

        # target scores: sum so backward computes grads for the full batch at once
        scores = logits.gather(1, class_idx_t.view(-1, 1)).squeeze(1)
        scores.sum().backward()

        acts = self._acts
        grads = self._grads
        if acts is None or grads is None:
            raise RuntimeError("No activations/gradients captured. Check target_layer selection.")

        # CNN featuremap case: (B,C,H,W) classic Grad-CAM
        if acts.ndim == 4:
            # channel weights = spatially averaged gradients
            weights = grads.mean(dim=(2, 3), keepdim=True)       # (B,C,1,1)
            cam = (weights * acts).sum(dim=1, keepdim=True)      # (B,1,H,W)
            cam = F.relu(cam)
            cam = self._normalize(cam)
            return CAMOutput(cam=cam.detach(), logits=logits.detach(), class_idx=class_idx_t.detach())

        # ViT token case: (B,N,D) token-level Grad-CAM-like
        if acts.ndim == 3:
            if vit_grid is None:
                raise ValueError("vit_grid=(H_patches,W_patches) must be provided for ViT token CAM.")

            H_p, W_p = vit_grid
            B2, N, D = acts.shape
            if B2 != B:
                raise RuntimeError("Batch size mismatch between logits and activations.")

            # timm ViT typically has CLS token => N == 1 + H_p*W_p
            if N == 1 + H_p * W_p:
                acts_p = acts[:, 1:, :]
                grads_p = grads[:, 1:, :]
            elif N == H_p * W_p:
                acts_p = acts
                grads_p = grads
            else:
                raise ValueError(
                    f"Token count N={N} doesn't match vit_grid={H_p}x{W_p} "
                    f"(expected {H_p*W_p} or {1+H_p*W_p})."
                )

            # token relevance: sum over embedding dim of grad * activation
            token_scores = (acts_p * grads_p).sum(dim=2)  # (B, H_p*W_p)
            token_scores = F.relu(token_scores)

            cam = token_scores.view(B, 1, H_p, W_p)
            cam = self._normalize(cam)
            return CAMOutput(cam=cam.detach(), logits=logits.detach(), class_idx=class_idx_t.detach())

        raise ValueError(f"Unsupported activation shape for CAM: {tuple(acts.shape)}")


def upsample_cam_to_image(cam: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Utility: upsample CAM (B,1,h,w) to (B,1,H,W) for overlay on 224x224 images.
    """
    return F.interpolate(cam, size=image_hw, mode="bilinear", align_corners=False)