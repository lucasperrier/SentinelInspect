"""Metric computation, separated from the script that writes files.

Pure functions over arrays: no IO, no config, no model. That is what makes the
tricky parts -- the weighted loss and the threshold tuning -- testable without
a checkpoint or a dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

CLASS_TARGET_NAMES = ["no_crack", "crack"]


def weighted_mean_loss(batch_losses: Sequence[float], batch_sizes: Sequence[int]) -> float:
    """Mean loss per *sample*, not per batch.

    Averaging batch means over-weights the final batch, which is usually
    smaller: with batch_size=64 and 5,889 samples the last batch holds 25
    samples but would count as much as a full one.
    """
    if not batch_losses:
        return float("nan")
    if len(batch_losses) != len(batch_sizes):
        raise ValueError("batch_losses and batch_sizes must be the same length")
    total = sum(int(n) for n in batch_sizes)
    if total == 0:
        return float("nan")
    pairs = zip(batch_losses, batch_sizes, strict=True)
    return float(sum(loss * size for loss, size in pairs) / total)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
    prefix: str = "test",
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        f"{prefix}_acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    # AUC is undefined when only one class is present. That specific case is
    # worth tolerating; a bare `except Exception` would also swallow real bugs.
    if len(np.unique(y_true)) < 2:
        metrics[f"{prefix}_auc"] = None
    else:
        metrics[f"{prefix}_auc"] = float(roc_auc_score(y_true, y_prob_pos))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = tn, fp, fn, tp

    # Recall on the positive class is the number that matters for inspection:
    # a missed crack is the expensive error, a false alarm merely costs a look.
    metrics[f"{prefix}_recall_crack"] = float(tp / (tp + fn)) if (tp + fn) else None
    metrics[f"{prefix}_precision_crack"] = float(tp / (tp + fp)) if (tp + fp) else None
    metrics[f"{prefix}_specificity"] = float(tn / (tn + fp)) if (tn + fp) else None

    # labels=[0, 1] is required: without it sklearn infers the classes present
    # and raises when a split contains only one, which is exactly the degenerate
    # case this function is supposed to survive.
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_TARGET_NAMES, digits=4, zero_division=0
    )
    return metrics


def review_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """What the triage rule actually buys.

    Two numbers decide whether a band is worth deploying: how much work it
    creates (review_rate) and how much of the model's error it intercepts
    (errors_caught_rate). A band that flags half the dataset to catch 60% of
    errors is not useful.
    """
    flagged = (y_prob_pos >= lower) & (y_prob_pos <= upper)
    errors = y_true != y_pred
    n = len(y_true)

    auto = ~flagged
    return {
        "review_lower": float(lower),
        "review_upper": float(upper),
        "review_count": int(flagged.sum()),
        "review_rate": float(flagged.mean()) if n else 0.0,
        "errors_total": int(errors.sum()),
        "errors_caught": int((errors & flagged).sum()),
        "errors_caught_rate": float((errors & flagged).sum() / errors.sum()) if errors.sum() else None,
        "auto_decided_count": int(auto.sum()),
        "auto_decided_accuracy": float((~errors[auto]).mean()) if auto.sum() else None,
    }


def tune_review_band(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    target_error_recall: float = 0.80,
    max_review_rate: float = 0.10,
    step: float = 0.01,
) -> tuple[float, float]:
    """Choose the narrowest symmetric band catching `target_error_recall` of errors.

    Tuned on validation, never on test. Widening the band always catches more
    errors, so an unconstrained search returns [0, 1] -- hence `max_review_rate`,
    which encodes the review capacity actually available.

    Returns the widest band tried if the target cannot be met inside the budget;
    the caller sees the shortfall in `review_statistics`.
    """
    y_pred = (y_prob_pos >= 0.5).astype(int)
    errors = y_true != y_pred
    total_errors = int(errors.sum())

    best = (0.5, 0.5)
    half = step
    while half <= 0.5 + 1e-9:
        lower, upper = 0.5 - half, 0.5 + half
        flagged = (y_prob_pos >= lower) & (y_prob_pos <= upper)
        review_rate = float(flagged.mean())
        if review_rate > max_review_rate:
            break
        best = (round(lower, 4), round(upper, 4))
        caught = float((errors & flagged).sum() / total_errors) if total_errors else 1.0
        if caught >= target_error_recall:
            return best
        half += step
    return best
