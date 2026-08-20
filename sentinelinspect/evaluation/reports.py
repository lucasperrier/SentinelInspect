"""Writing an evaluation bundle to disk.

Separated from metric computation so the numbers can be tested without a
filesystem, and so the bundle's shape is defined in one place rather than
scattered through the evaluation script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

BUNDLE_FILES = (
    "metrics.json",
    "classification_report.txt",
    "confusion_matrix.npy",
    "predictions.npz",
)


def write_bundle(
    output_dir: str | Path,
    metrics: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
    needs_review: Optional[np.ndarray] = None,
    relative_paths: Optional[Sequence[str]] = None,
) -> Path:
    """Write the artifacts that define an evaluation result.

    `predictions.npz` holds per-sample outputs so any new metric -- a different
    threshold, a per-substrate breakdown -- can be computed later without
    re-running the model. That is the reason to keep it rather than metrics
    alone.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    serialisable = {k: v for k, v in metrics.items()}
    (out / "metrics.json").write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    report = metrics.get("classification_report", "")
    (out / "classification_report.txt").write_text(report, encoding="utf-8")

    np.save(out / "confusion_matrix.npy", np.array(metrics["confusion_matrix"], dtype=np.int64))

    arrays: Dict[str, np.ndarray] = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob_pos": np.asarray(y_prob_pos),
    }
    if needs_review is not None:
        arrays["needs_review"] = np.asarray(needs_review)
    if relative_paths is not None:
        arrays["relative_path"] = np.asarray(list(relative_paths), dtype=object)
    np.savez(out / "predictions.npz", **arrays)

    return out


def summarise(metrics: Dict[str, Any], review: Optional[Dict[str, Any]] = None) -> str:
    """A few lines a human can read without opening the JSON."""
    lines = [
        f"accuracy          {metrics.get('test_acc', float('nan')):.4f}",
        f"f1                {metrics.get('test_f1', float('nan')):.4f}",
    ]
    auc = metrics.get("test_auc")
    lines.append(f"auc               {auc:.4f}" if auc is not None else "auc               n/a")
    recall = metrics.get("test_recall_crack")
    if recall is not None:
        lines.append(f"recall (crack)    {recall:.4f}")
    lines.append(f"confusion         tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}")

    if review:
        lines.append("")
        lines.append(f"review band       [{review['review_lower']:.2f}, {review['review_upper']:.2f}]")
        lines.append(f"routed to review  {review['review_count']} ({review['review_rate']:.1%})")
        caught = review.get("errors_caught_rate")
        if caught is not None:
            lines.append(
                f"errors caught     {review['errors_caught']}/{review['errors_total']} ({caught:.1%})"
            )
        auto_acc = review.get("auto_decided_accuracy")
        if auto_acc is not None:
            lines.append(f"auto-decided acc  {auto_acc:.4f} on {review['auto_decided_count']} images")
    return "\n".join(lines)
