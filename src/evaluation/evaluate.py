from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from src.config.load import to_runtime_config
from src.data.datamodule import CrackDataModule
from src.models.resnet50 import ResNet50Module
from src.models.vit import VisionTransformerModule


def build_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    name = str(model_cfg.get("name", "resnet50")).lower()
    if "vit" in name:
        return VisionTransformerModule(model_cfg)
    return ResNet50Module(model_cfg)


@torch.no_grad()
def predict_on_test(model: torch.nn.Module, datamodule: CrackDataModule, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    model.to(device)
    dl = datamodule.test_dataloader()
    ce = torch.nn.CrossEntropyLoss()

    y_true_all, y_pred_all, y_prob_all, losses = [], [], [], []

    for x, y in dl:
        x = x.to(device)
        y = torch.as_tensor(y, device=device).long()
        logits = model(x)
        loss = ce(logits, y)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())
        y_prob_all.append(probs[:, 1].cpu().numpy())
        losses.append(loss.item())

    return (
        np.concatenate(y_true_all, axis=0),
        np.concatenate(y_pred_all, axis=0),
        np.concatenate(y_prob_all, axis=0),
        float(np.mean(losses)) if losses else float("nan"),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, average="binary")),
    }
    try:
        metrics["test_auc"] = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        metrics["test_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = int(tn), int(fp), int(fn), int(tp)
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=["no_crack", "crack"], digits=4, zero_division=0
    )
    return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    runtime = to_runtime_config(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if runtime.trainer and runtime.trainer.deterministic:
        seed_everything(runtime.seed, workers=True)

    output_dir = Path("reports/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = CrackDataModule(
            batch_size=runtime.data.batch_size,
            num_workers=runtime.data.num_workers,
            preprocessing={"image_size": runtime.data.image_size},
            manifest_path=runtime.data.manifest_path,
            train_split_path=runtime.data.train_split_path,
            val_split_path=runtime.data.val_split_path,
            test_split_path=runtime.data.test_split_path,
            raw_root=runtime.data.raw_root,
            validate_artifacts=runtime.data.validate_artifacts,
            fail_on_validation_error=runtime.data.fail_on_validation_error,
    )

    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    
    model = build_model(cfg_dict["model"])
    if runtime.checkpoint_path:
        model_cls = VisionTransformerModule if "vit" in runtime.model.name else ResNet50Module
        model = model_cls.load_from_checkpoint(runtime.checkpoint_path, config=cfg_dict["model"])

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator=runtime.trainer.accelerator if runtime.trainer else "auto",
        deterministic=runtime.trainer.deterministic if runtime.trainer else True,
    )
    trainer.test(model, datamodule=datamodule, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred, y_prob_pos, avg_loss = predict_on_test(model, datamodule, device)
    metrics = compute_metrics(y_true, y_pred, y_prob_pos)
    metrics["test_loss"] = avg_loss
    metrics["checkpoint_path"] = runtime.checkpoint_path

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    np.save(output_dir / "confusion_matrix.npy", np.array(metrics["confusion_matrix"], dtype=np.int64))
    np.savez(output_dir / "predictions.npz", y_true=y_true, y_pred=y_pred, y_prob_pos=y_prob_pos)

    print(f"Saved evaluation artifacts to: {output_dir}")


if __name__ == "__main__":
    main()