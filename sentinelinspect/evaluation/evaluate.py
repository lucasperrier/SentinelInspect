"""Offline evaluation.

Scores a checkpoint on one split and writes a report bundle. Predictions come
from the same `Predictor` the CLI and the API use, so an offline metric and a
served decision cannot disagree.

    python -m sentinelinspect.evaluation.evaluate \
        "checkpoint_path='runs/<exp>/<model>.ckpt'" split=test
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset

from sentinelinspect.config.load import to_runtime_config
from sentinelinspect.data.datamodule import CrackDataModule
from sentinelinspect.evaluation.metrics import (
    compute_metrics,
    review_statistics,
    tune_review_band,
    weighted_mean_loss,
)
from sentinelinspect.evaluation.reports import summarise, write_bundle
from sentinelinspect.inference.contracts import POSITIVE_INDEX
from sentinelinspect.inference.predictor import Predictor

EPS = 1e-12


def select_dataset(datamodule: CrackDataModule, split: str) -> Dataset:
    datasets = {
        "train": datamodule.train_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }
    if split not in datasets:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(datasets)}")
    dataset = datasets[split]
    if dataset is None:
        raise RuntimeError(f"Split {split!r} was not built. Did setup() run?")
    return dataset


def score_split(
    predictor: Predictor,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One pass over the data. Shuffle is off so rows stay aligned with the
    dataset's path list, which the bundle records alongside the predictions."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    y_true: list[np.ndarray] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    review: list[bool] = []
    batch_losses: list[float] = []
    batch_sizes: list[int] = []

    for images, labels in loader:
        predictions = predictor.predict_tensor(images)
        probs_pos = np.array([p.probabilities["crack"] for p in predictions])
        labels_np = labels.numpy()

        y_true.append(labels_np)
        y_pred.extend(p.predicted_index for p in predictions)
        y_prob.extend(probs_pos.tolist())
        review.extend(p.needs_review for p in predictions)

        # cross-entropy from the probability of the true class; collected per
        # batch and combined by sample count, never as a mean of batch means
        true_probs = np.where(labels_np == POSITIVE_INDEX, probs_pos, 1.0 - probs_pos)
        batch_losses.append(float(-np.log(np.clip(true_probs, EPS, 1.0)).mean()))
        batch_sizes.append(len(labels_np))

    return (
        np.concatenate(y_true) if y_true else np.array([]),
        np.array(y_pred),
        np.array(y_prob),
        np.array(review),
        weighted_mean_loss(batch_losses, batch_sizes),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    runtime = to_runtime_config(cfg)

    if runtime.trainer and runtime.trainer.deterministic:
        seed_everything(runtime.seed, workers=True)

    if not runtime.checkpoint_path:
        raise SystemExit(
            "checkpoint_path is required for evaluation. Without it this scored a "
            "randomly initialised model and wrote the result out as a metric."
        )

    split = runtime.split or "test"
    output_dir = Path(runtime.output_dir or f"reports/eval_{split}")

    datamodule = CrackDataModule(
        batch_size=runtime.data.batch_size,
        num_workers=runtime.data.num_workers,
        preprocessing=runtime.preprocessing.model_dump() if runtime.preprocessing else None,
        manifest_path=runtime.data.manifest_path,
        train_split_path=runtime.data.train_split_path,
        val_split_path=runtime.data.val_split_path,
        test_split_path=runtime.data.test_split_path,
        raw_root=runtime.data.raw_root,
        validate_artifacts=runtime.data.validate_artifacts,
        fail_on_validation_error=runtime.data.fail_on_validation_error,
    )
    datamodule.setup(stage="test")
    dataset = select_dataset(datamodule, split)

    predictor = Predictor.from_runtime_config(runtime)
    y_true, y_pred, y_prob_pos, needs_review, loss = score_split(
        predictor, dataset, runtime.data.batch_size, runtime.data.num_workers
    )

    metrics = compute_metrics(y_true, y_pred, y_prob_pos, prefix="test")
    metrics["test_loss"] = loss
    metrics["split"] = split
    metrics["checkpoint_path"] = runtime.checkpoint_path
    metrics["checkpoint_sha256"] = predictor.metadata.checkpoint_sha256
    metrics["n_samples"] = int(len(y_true))

    policy = predictor.review_policy
    review = review_statistics(y_true, y_pred, y_prob_pos, policy.lower, policy.upper)
    metrics.update(review)

    # The band is chosen on validation and then held fixed. Tuning it on test
    # would be selecting a hyperparameter on the set used to report the result.
    if split == "val":
        suggested_lower, suggested_upper = tune_review_band(y_true, y_prob_pos)
        metrics["suggested_review_lower"] = suggested_lower
        metrics["suggested_review_upper"] = suggested_upper

    paths = getattr(dataset, "image_paths", None)
    relative = [str(Path(p).name) for p in paths] if paths else None
    write_bundle(output_dir, metrics, y_true, y_pred, y_prob_pos, needs_review, relative)

    print(f"\n=== {split} ===")
    print(summarise(metrics, review))
    if split == "val":
        print(
            f"\nsuggested band    [{metrics['suggested_review_lower']:.2f}, "
            f"{metrics['suggested_review_upper']:.2f}]  -> write into configs/review/default.yaml"
        )
    print(f"\nBundle written to: {output_dir}")


if __name__ == "__main__":
    main()
