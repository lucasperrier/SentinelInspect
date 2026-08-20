"""Turning a checkpoint on disk into a model in memory.

Separated from the Predictor so that "how do I load this?" has exactly one
answer, and so the service can load once at startup rather than per request.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from sentinelinspect.data.build_manifest import sha256_file
from sentinelinspect.models.base import CrackClassifier
from sentinelinspect.models.factory import model_class_for


class CheckpointNotFoundError(FileNotFoundError):
    """Raised instead of a bare FileNotFoundError so callers -- especially the
    API's startup hook -- can distinguish a missing model from any other IO
    failure and report it as a configuration problem."""


def checkpoint_fingerprint(checkpoint_path: str | Path, length: int = 16) -> str:
    """Short content digest of a checkpoint file.

    Goes into every prediction's metadata. A filename is not enough to identify
    weights -- two runs happily produce the same name -- so a stored prediction
    would otherwise be impossible to trace back to the model that made it.
    """
    return sha256_file(Path(checkpoint_path))[:length]


def resolve_checkpoint(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")
    if not path.is_file():
        raise CheckpointNotFoundError(f"Checkpoint path is not a file: {path}")
    return path


# Which architecture a checkpoint *is* is a property of the checkpoint, not of
# whatever config file happens to be on disk at load time.
ARCHITECTURE_KEYS = ("name", "model", "num_classes")


def read_checkpoint_hyperparameters(checkpoint_path: str | Path) -> dict[str, Any]:
    """The hyper_parameters Lightning stored when the checkpoint was written."""
    payload = torch.load(str(resolve_checkpoint(checkpoint_path)), map_location="cpu", weights_only=False)
    stored = payload.get("hyper_parameters", {}) or {}
    return dict(stored)


def resolve_model_config(
    checkpoint_path: str | Path,
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge a config with the checkpoint, letting the checkpoint win on architecture.

    Config supplies everything else; the checkpoint decides what it is. Trusting
    the config instead is how you get a wall of shape-mismatch errors at deploy
    time after someone edits `configs/model/*.yaml` -- the weights on disk did
    not change, only the file claiming to describe them did.
    """
    config: dict[str, Any] = dict(model_config or {})
    stored = read_checkpoint_hyperparameters(checkpoint_path)

    for key in ARCHITECTURE_KEYS:
        if key in stored:
            requested = config.get(key)
            if requested is not None and requested != stored[key]:
                warnings.warn(
                    f"Config requested {key}={requested!r} but the checkpoint was trained "
                    f"with {key}={stored[key]!r}. Using the checkpoint's value.",
                    stacklevel=2,
                )
            config[key] = stored[key]

    # the checkpoint carries every weight already; fetching ImageNet weights
    # first would download tens of megabytes only to overwrite them
    config["pretrained"] = False
    return config


def load_model(
    checkpoint_path: str | Path,
    model_config: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> CrackClassifier:
    """Load a trained model, ready for inference."""
    path = resolve_checkpoint(checkpoint_path)

    config = resolve_model_config(path, model_config)
    model_cls = model_class_for(config.get("name", ""))
    model = model_cls.load_from_checkpoint(str(path), config=config, map_location=device)

    model.eval()
    model.to(device)
    # inference only: no autograd graph, no accidental weight updates
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick a device, honouring an explicit preference but never inventing a GPU."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("device=gpu was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
