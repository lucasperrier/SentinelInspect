from pathlib import Path

import numpy as np
import pytest
import torch

from src.preprocessing.transforms import (
    DEFAULTS,
    build_eval_transforms,
    build_inference_transforms,
    build_train_transforms,
    build_val_transforms,
    resolve_preprocessing,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _dummy_image(h=300, w=400, c=3):
    return np.random.randint(0, 256, size=(h, w, c), dtype=np.uint8)


def test_train_transforms_output_tensor_shape_and_dtype():
    t = build_train_transforms({"image_size": 224})
    out = t(image=_dummy_image())["image"]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_val_eval_inference_have_same_deterministic_output():
    """Evaluation and serving must not diverge.

    If these ever differ, the model is scored on one distribution and served
    another, and the gap shows up as unexplained production drift.
    """
    pre = {"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    img = _dummy_image()

    val_out = build_val_transforms(pre)(image=img)["image"]
    eval_out = build_eval_transforms(pre)(image=img)["image"]
    inf_out = build_inference_transforms(pre)(image=img)["image"]

    assert torch.allclose(val_out, eval_out)
    assert torch.allclose(val_out, inf_out)
    assert val_out.shape == (3, 224, 224)


def test_defaults_apply_when_nothing_is_passed():
    out = build_val_transforms(None)(image=_dummy_image())["image"]
    assert out.shape == (3, 224, 224)


def test_settings_actually_reach_the_pipeline():
    """The bug this contract change fixes.

    The old signature expected a dict containing a `preprocessing` key, so a
    caller passing the block itself got silently ignored and fell back to
    defaults. Asserting on a non-default value is what makes that visible.
    """
    t = build_val_transforms({"image_size": 64})
    out = t(image=_dummy_image())["image"]
    assert out.shape == (3, 64, 64), "image_size was ignored"


def test_unknown_preprocessing_key_is_rejected():
    """A typo must fail loudly rather than silently doing nothing."""
    with pytest.raises(ValueError, match="hflip_prob"):
        resolve_preprocessing({"hflip_prob": 0.5})


def test_resolve_fills_every_default():
    resolved = resolve_preprocessing({"image_size": 128})
    assert resolved["image_size"] == 128
    assert set(resolved) == set(DEFAULTS)


def test_hydra_config_reaches_the_transform():
    """End-to-end over the real config files.

    Every unit test above passes a dict written by the test. This one starts
    from configs/train.yaml, which is what production actually reads -- the
    gap that let the original defect survive a green suite.
    """
    from hydra import compose, initialize_config_dir

    from src.config.load import to_runtime_config

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        runtime = to_runtime_config(compose(config_name="train"))

    assert runtime.preprocessing is not None
    pre = runtime.preprocessing.model_dump()

    # values only present if the real YAML was read, not the hardcoded defaults
    assert "hflip_p" in pre
    assert "shift_limit" in pre

    train_t = build_train_transforms(pre)
    names = [type(step).__name__ for step in train_t.transforms]
    assert names == [
        "Resize",
        "HorizontalFlip",
        "RandomBrightnessContrast",
        "Affine",
        "Normalize",
        "ToTensorV2",
    ]

    out = train_t(image=_dummy_image())["image"]
    assert out.shape == (3, pre["image_size"], pre["image_size"])
