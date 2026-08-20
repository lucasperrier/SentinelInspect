"""Shared fixtures.

Note what is *not* here any more: the sys.path insert that used to make
`import sentinelinspect` work. The package is installed, so that hack is gone.

The checkpoint fixture builds a valid Lightning checkpoint by saving a
state_dict directly rather than by training. Training even a tiny model in a
test suite makes the suite slow enough that people stop running it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sentinelinspect.models.factory import build_model

TINY_MODEL_CONFIG = {
    "name": "resnet50",       # registry key -> ResNet50Module
    "model": "resnet18",      # small timm backbone, no download needed
    "num_classes": 2,
    "pretrained": False,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "scheduler": None,
}


@pytest.fixture(scope="session")
def model_config() -> dict:
    return dict(TINY_MODEL_CONFIG)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, model_config) -> Path:
    """A real, loadable Lightning checkpoint with untrained weights.

    Untrained is fine: these tests assert on the *contract*, not on accuracy.
    """
    model = build_model(model_config)
    path = tmp_path_factory.mktemp("checkpoints") / "tiny.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model_config),
            "pytorch-lightning_version": "2.0.0",
            "epoch": 0,
            "global_step": 0,
        },
        path,
    )
    return path


@pytest.fixture(scope="session")
def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(
        np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    ).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def image_path(tmp_path_factory, image_bytes) -> Path:
    path = tmp_path_factory.mktemp("images") / "sample.jpg"
    path.write_bytes(image_bytes)
    return path


@pytest.fixture(scope="session")
def predictor(tiny_checkpoint, model_config):
    from sentinelinspect.inference.predictor import Predictor

    return Predictor.from_checkpoint(
        checkpoint_path=tiny_checkpoint,
        model_config=model_config,
        preprocessing={"image_size": 64},
        device="cpu",
    )
