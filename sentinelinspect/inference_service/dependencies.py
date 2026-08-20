"""Wiring: configuration, and the single Predictor the service shares.

The model is loaded once at startup and handed to every request. Loading per
request would add seconds of latency and defeat the point of the shared core.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException, Request, status

from sentinelinspect.config.load import to_runtime_config
from sentinelinspect.config.schema import RuntimeConfig
from sentinelinspect.inference.predictor import Predictor

# configs/ lives at the repository root, next to the package. The container
# copies it to the same relative position.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

ENV_CHECKPOINT = "SENTINELINSPECT_CHECKPOINT"
ENV_CONFIG_DIR = "SENTINELINSPECT_CONFIG_DIR"
ENV_MAX_UPLOAD_MB = "SENTINELINSPECT_MAX_UPLOAD_MB"

DEFAULT_MAX_UPLOAD_MB = 10


def max_upload_bytes() -> int:
    return int(float(os.getenv(ENV_MAX_UPLOAD_MB, DEFAULT_MAX_UPLOAD_MB)) * 1024 * 1024)


def load_runtime_config(
    checkpoint_path: Optional[str] = None,
    config_dir: Optional[str | Path] = None,
    overrides: Optional[List[str]] = None,
) -> RuntimeConfig:
    """Compose the same Hydra config the CLI uses.

    The service deliberately does not invent its own settings format: one config
    system means the band the API applies is the band evaluation measured.
    """
    from hydra import compose, initialize_config_dir

    directory = Path(config_dir or os.getenv(ENV_CONFIG_DIR) or DEFAULT_CONFIG_DIR).resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Config directory not found: {directory}")

    resolved_overrides = list(overrides or [])
    checkpoint = checkpoint_path or os.getenv(ENV_CHECKPOINT)
    if checkpoint:
        # quoted: checkpoint filenames contain '=', which Hydra parses as a separator
        resolved_overrides.append(f"checkpoint_path='{checkpoint}'")

    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(config_name="inference", overrides=resolved_overrides)
    return to_runtime_config(cfg)


def build_predictor(
    checkpoint_path: Optional[str] = None,
    config_dir: Optional[str | Path] = None,
) -> Predictor:
    runtime = load_runtime_config(checkpoint_path=checkpoint_path, config_dir=config_dir)
    if not runtime.checkpoint_path:
        raise RuntimeError(
            f"No checkpoint configured. Set {ENV_CHECKPOINT} or pass checkpoint_path."
        )
    return Predictor.from_runtime_config(runtime)


def get_predictor(request: Request) -> Predictor:
    """FastAPI dependency: the Predictor built during startup.

    A 503 rather than a 500 if it is absent -- the service is up but not ready,
    which is a different thing from a crash and is what a load balancer needs
    to distinguish.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded; the service is not ready.",
        )
    return predictor
