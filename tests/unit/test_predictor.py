from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sentinelinspect.inference.contracts import Prediction, ReviewPolicy
from sentinelinspect.inference.model_loader import (
    CheckpointNotFoundError,
    checkpoint_fingerprint,
    load_model,
)
from sentinelinspect.inference.predictor import InvalidImageError, Predictor


def test_prediction_satisfies_the_documented_contract(predictor, image_path):
    p = predictor.predict_image(image_path)

    assert isinstance(p, Prediction)
    assert p.predicted_label in ("crack", "no_crack")
    assert 0.0 <= p.confidence_score <= 1.0
    assert set(p.probabilities) == {"crack", "no_crack"}
    assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert isinstance(p.needs_review, bool)
    assert p.model_metadata.checkpoint_sha256
    assert p.latency_ms >= 0.0


def test_confidence_is_the_probability_of_the_predicted_class(predictor, image_path):
    p = predictor.predict_image(image_path)
    assert p.confidence_score == pytest.approx(p.probabilities[p.predicted_label])


def test_every_input_form_gives_the_same_answer(predictor, image_path, image_bytes):
    """A path, raw bytes, a PIL image and an array are four ways to hand over
    the same pixels. If they disagree, the API and the CLI disagree."""
    pil = Image.open(image_path).convert("RGB")
    results = [
        predictor.predict_image(image_path),
        predictor.predict_image(str(image_path)),
        predictor.predict_image(image_bytes),
        predictor.predict_image(pil),
        predictor.predict_image(np.array(pil)),
    ]
    first = results[0].probabilities["crack"]
    for r in results[1:]:
        assert r.probabilities["crack"] == pytest.approx(first, abs=1e-6)


def test_image_path_and_tensor_paths_agree(predictor, image_path):
    """Offline evaluation feeds tensors; serving feeds images. Both go through
    _probabilities, so the two must produce identical numbers -- this is the
    train/serve skew regression test."""
    from_image = predictor.predict_image(image_path)

    array = np.array(Image.open(image_path).convert("RGB"))
    tensor = predictor.transform(image=array)["image"].unsqueeze(0)
    from_tensor = predictor.predict_tensor(tensor)[0]

    assert from_tensor.probabilities["crack"] == pytest.approx(
        from_image.probabilities["crack"], abs=1e-6
    )
    assert from_tensor.predicted_label == from_image.predicted_label


def test_batch_matches_one_at_a_time(predictor, image_path):
    batch = predictor.predict_images([image_path, image_path, image_path])
    single = predictor.predict_image(image_path)

    assert len(batch.predictions) == 3
    for p in batch.predictions:
        assert p.probabilities["crack"] == pytest.approx(single.probabilities["crack"], abs=1e-6)


def test_empty_batch_is_not_an_error(predictor):
    batch = predictor.predict_images([])
    assert batch.predictions == []
    assert batch.review_rate == 0.0


def test_corrupt_bytes_raise_invalid_image(predictor):
    with pytest.raises(InvalidImageError, match="decode"):
        predictor.predict_image(b"this is definitely not a jpeg")


def test_missing_image_raises_invalid_image(predictor, tmp_path):
    with pytest.raises(InvalidImageError, match="not found"):
        predictor.predict_image(tmp_path / "nope.jpg")


def test_unsupported_input_type_raises_invalid_image(predictor):
    with pytest.raises(InvalidImageError, match="Unsupported"):
        predictor.predict_image(12345)


def test_missing_checkpoint_fails_with_a_typed_error(model_config, tmp_path):
    """A distinct exception type so the API can answer 'misconfigured' rather
    than surfacing a bare FileNotFoundError as a 500."""
    with pytest.raises(CheckpointNotFoundError, match="not found"):
        load_model(tmp_path / "absent.ckpt", model_config)


def test_directory_instead_of_checkpoint_is_rejected(model_config, tmp_path):
    with pytest.raises(CheckpointNotFoundError, match="not a file"):
        load_model(tmp_path, model_config)


def test_fingerprint_is_stable_and_content_addressed(tiny_checkpoint, tmp_path):
    copy = tmp_path / "renamed.ckpt"
    copy.write_bytes(Path(tiny_checkpoint).read_bytes())

    assert checkpoint_fingerprint(tiny_checkpoint) == checkpoint_fingerprint(copy)
    assert len(checkpoint_fingerprint(tiny_checkpoint)) == 16


def test_model_is_loaded_once_not_per_call(predictor, image_path):
    """The whole reason the service holds a Predictor instead of calling a
    module-level predict() per request."""
    before = predictor.model
    predictor.predict_image(image_path)
    predictor.predict_image(image_path)
    assert predictor.model is before


def test_model_is_in_eval_mode_with_grad_disabled(predictor):
    """Dropout and batchnorm behave differently in train mode; leaving a served
    model in train mode makes predictions non-deterministic."""
    assert predictor.model.training is False
    assert all(not p.requires_grad for p in predictor.model.parameters())


def test_review_policy_is_applied_to_the_prediction(tiny_checkpoint, model_config, image_path):
    """A band covering everything must flag everything; a zero-width band at 0
    must flag (almost) nothing. Proves the policy is wired in, not decorative."""
    always = Predictor.from_checkpoint(
        tiny_checkpoint, model_config, preprocessing={"image_size": 64},
        review_policy=ReviewPolicy(lower=0.0, upper=1.0),
    )
    never = Predictor.from_checkpoint(
        tiny_checkpoint, model_config, preprocessing={"image_size": 64},
        review_policy=ReviewPolicy(lower=0.0, upper=0.0),
    )
    assert always.predict_image(image_path).needs_review is True
    assert never.predict_image(image_path).needs_review is False


def test_predict_tensor_rejects_an_unbatched_tensor(predictor):
    with pytest.raises(ValueError, match="4D"):
        predictor.predict_tensor(torch.randn(3, 64, 64))
