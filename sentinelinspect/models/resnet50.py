from __future__ import annotations

import timm
import torch.nn as nn

from sentinelinspect.models.base import CrackClassifier


class ResNet50Module(CrackClassifier):
    """ResNet-50 backbone. Everything but the backbone lives in CrackClassifier."""

    default_learning_rate = 1e-3
    default_weight_decay = 1e-4
    default_optimizer = "adam"

    def build_backbone(self) -> nn.Module:
        model = timm.create_model(
            self.hparams.get("model", "resnet50"),
            pretrained=self.hparams.get("pretrained", True),
            num_classes=self.hparams.get("num_classes", 2),
        )
        if self.hparams.get("freeze_backbone", False):
            for param in model.parameters():
                param.requires_grad = False
            classifier = model.get_classifier()
            if isinstance(classifier, nn.Module):
                for param in classifier.parameters():
                    param.requires_grad = True
        return model
