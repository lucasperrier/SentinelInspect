from __future__ import annotations

import timm
import torch
import torch.nn as nn

from sentinelinspect.models.base import CrackClassifier


class VisionTransformerModule(CrackClassifier):
    """ViT backbone, plus the regularisation and freezing knobs timm exposes."""

    default_learning_rate = 1e-4
    default_weight_decay = 1e-5
    default_optimizer = "adamw"

    def build_backbone(self) -> nn.Module:
        model = timm.create_model(
            self.hparams.get("model", "vit_base_patch16_224"),
            pretrained=self.hparams.get("pretrained", True),
            num_classes=self.hparams.get("num_classes", 2),
            drop_rate=float(self.hparams.get("drop_rate", 0.0)),
            drop_path_rate=float(self.hparams.get("drop_path_rate", 0.0)),
        )
        self._apply_freezing(model)
        return model

    def build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(label_smoothing=float(self.hparams.get("label_smoothing", 0.0)))

    def _apply_freezing(self, model: nn.Module) -> None:
        freeze_mode = str(self.hparams.get("freeze_mode", "first_n_blocks")).lower()
        freeze_layers = int(self.hparams.get("freeze_layers", 0))

        if freeze_mode == "head_only":
            for p in model.parameters():
                p.requires_grad = False
            for name in ("head", "fc", "classifier"):
                module = getattr(model, name, None)
                if isinstance(module, nn.Module):
                    for p in module.parameters():
                        p.requires_grad = True
            return

        if not freeze_layers:
            return

        # freeze the stem as well: common when fine-tuning on little data
        for name in ("patch_embed", "pos_embed", "cls_token", "norm_pre", "norm"):
            obj = getattr(model, name, None)
            if isinstance(obj, torch.nn.Parameter):
                obj.requires_grad = False
            elif isinstance(obj, nn.Module):
                for p in obj.parameters():
                    p.requires_grad = False

        for idx, block in enumerate(getattr(model, "blocks", [])):
            if idx < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False
