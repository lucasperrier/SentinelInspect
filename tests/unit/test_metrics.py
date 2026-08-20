import numpy as np
import pytest

from sentinelinspect.evaluation.metrics import (
    compute_metrics,
    review_statistics,
    tune_review_band,
    weighted_mean_loss,
)


def test_weighted_loss_differs_from_a_mean_of_batch_means():
    """The bug this function replaces.

    Two batches: 64 samples at loss 0.1, then a final batch of 1 at loss 10.0.
    Averaging the batch means gives 5.05; weighting by samples gives 0.252.
    The old code reported the former.
    """
    losses, sizes = [0.1, 10.0], [64, 1]

    naive = sum(losses) / len(losses)
    weighted = weighted_mean_loss(losses, sizes)

    assert naive == pytest.approx(5.05)
    assert weighted == pytest.approx((0.1 * 64 + 10.0 * 1) / 65)
    assert weighted < naive


def test_weighted_loss_equals_naive_when_batches_are_equal():
    assert weighted_mean_loss([1.0, 3.0], [10, 10]) == pytest.approx(2.0)


def test_weighted_loss_handles_no_batches():
    assert np.isnan(weighted_mean_loss([], []))


def test_weighted_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        weighted_mean_loss([1.0, 2.0], [10])


def test_compute_metrics_on_a_known_confusion_matrix():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.8, 0.7, 0.4])

    m = compute_metrics(y_true, y_pred, y_prob)

    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert m["test_acc"] == pytest.approx(6 / 8)
    assert m["test_recall_crack"] == pytest.approx(3 / 4)
    assert m["test_precision_crack"] == pytest.approx(3 / 4)
    assert m["test_specificity"] == pytest.approx(3 / 4)


def test_auc_is_none_when_only_one_class_is_present():
    """Undefined, not an error -- but only this case is tolerated. A bare
    `except Exception` here would also hide genuine bugs."""
    y_true = np.array([1, 1, 1])
    m = compute_metrics(y_true, np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
    assert m["test_auc"] is None


def test_review_statistics_counts_work_and_errors_caught():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])          # two errors, at index 1 and 3
    y_prob = np.array([0.10, 0.55, 0.95, 0.45])

    stats = review_statistics(y_true, y_pred, y_prob, lower=0.4, upper=0.6)

    assert stats["review_count"] == 2         # 0.55 and 0.45 fall in the band
    assert stats["review_rate"] == pytest.approx(0.5)
    assert stats["errors_total"] == 2
    assert stats["errors_caught"] == 2        # both errors were uncertain
    assert stats["errors_caught_rate"] == pytest.approx(1.0)
    assert stats["auto_decided_accuracy"] == pytest.approx(1.0)


def test_review_statistics_with_no_errors():
    y = np.array([0, 1])
    stats = review_statistics(y, y, np.array([0.01, 0.99]), 0.4, 0.6)
    assert stats["errors_total"] == 0
    assert stats["errors_caught_rate"] is None


def test_tuned_band_respects_the_review_budget():
    """Widening always catches more errors, so an unconstrained search would
    return [0, 1]. The budget is what makes the answer meaningful."""
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, 2000)
    y_true = (y_prob > 0.5).astype(int)

    lower, upper = tune_review_band(y_true, y_prob, target_error_recall=0.99, max_review_rate=0.05)

    flagged = ((y_prob >= lower) & (y_prob <= upper)).mean()
    assert flagged <= 0.05 + 1e-9
    assert lower <= 0.5 <= upper


def test_tuned_band_is_narrow_when_the_model_is_confident():
    """A well-separated model needs almost no review: errors, if any, sit near
    the boundary and a thin band catches them."""
    y_true = np.array([0] * 100 + [1] * 100)
    y_prob = np.concatenate([np.full(100, 0.02), np.full(100, 0.98)])

    lower, upper = tune_review_band(y_true, y_prob, target_error_recall=0.8, max_review_rate=0.1)
    assert (upper - lower) < 0.1
