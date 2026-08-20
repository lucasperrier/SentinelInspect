from __future__ import annotations

from typing import Any, Dict, Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, F1Score

STAGES = ("train", "val", "test")


def _build_stage_metrics() -> nn.ModuleDict:
    """One fresh set of metrics for one stage.

    Torchmetrics objects accumulate state across batches, so each stage needs
    its own instances. Sharing one Accuracy between train and val would mix
    the two into a single meaningless number.
    """
    return nn.ModuleDict(
        {
            "acc": Accuracy(task="binary"),
            "f1": F1Score(task="binary"),
            "auc": AUROC(task="binary"),
        }
    )


class CrackClassifier(pl.LightningModule):
    """Training behaviour shared by every backbone.

    Subclasses supply only `build_backbone`. Metrics, the shared step, logging
    and optimiser construction live here so the model files cannot drift apart
    the way `resnet50.py` and `vit.py` did.
    """

    default_learning_rate: float = 1e-3
    default_weight_decay: float = 1e-4
    default_optimizer: str = "adam"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.save_hyperparameters(dict(config))

        self.learning_rate = float(self.hparams.get("learning_rate", self.default_learning_rate))
        self.weight_decay = float(self.hparams.get("weight_decay", self.default_weight_decay))
        self.optimizer_name = str(self.hparams.get("optimizer", self.default_optimizer)).lower()
        self.scheduler_name = self.hparams.get("scheduler")

        self.model = self.build_backbone()
        self.criterion = self.build_criterion()
        # NOTE: ModuleDict keys become attribute names on the ModuleDict, and
        # nn.Module already defines .train() and .eval(). Keying this by the bare
        # stage name raises KeyError: attribute 'train' already exists.
        self.metrics = nn.ModuleDict(
            {self._metrics_key(stage): _build_stage_metrics() for stage in STAGES}
        )

    # ---- subclass hooks -------------------------------------------------

    def build_backbone(self) -> nn.Module:
        raise NotImplementedError("Subclasses must build their own backbone")

    def build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    # ---- shared behaviour -----------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _metrics_key(stage: str) -> str:
        return f"{stage}_metrics"

    def stage_metrics(self, stage: str) -> nn.ModuleDict:
        key = self._metrics_key(stage)
        if key not in self.metrics:
            raise ValueError(f"Unknown stage {stage!r}. Expected one of {STAGES}")
        return self.metrics[key]

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        stage_metrics = self.stage_metrics(stage)

        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        # accuracy and F1 score hard decisions; AUROC needs a continuous score
        # to sweep a threshold over, so it gets the probability instead
        stage_metrics["acc"](preds, y)
        stage_metrics["f1"](preds, y)
        stage_metrics["auc"](probs, y)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage != "train"))
        # passing the metric object, not a number: Lightning calls compute()
        # at epoch end and reset() afterwards
        self.log(f"{stage}_acc", stage_metrics["acc"], prog_bar=True)
        self.log(f"{stage}_f1", stage_metrics["f1"])
        self.log(f"{stage}_auc", stage_metrics["auc"])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(
                f"Unsupported optimizer {self.optimizer_name!r}. Expected adam, adamw or sgd."
            )

        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(self.hparams.get("epochs", 10))
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if self.scheduler_name not in (None, "none"):
            raise ValueError(f"Unsupported scheduler {self.scheduler_name!r}. Expected 'cosine' or null.")
        return optimizer

    # ---- freezing helpers ------------------------------------------------

    def _freeze_all_but_head(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for name in ("head", "fc", "classifier"):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    p.requires_grad = True
