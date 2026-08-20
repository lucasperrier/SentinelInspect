"""Locating the Hydra config directory.

`@hydra.main(config_path="../../configs")` resolves relative to the file that
declares it, which works when you run `python -m sentinelinspect...` from the
repository but not through an installed console script -- there Hydra falls
back to module-based resolution and reports "Primary config module 'configs'
not found".

Computing an absolute path from the package location fixes that, and an
environment variable lets a container point somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_CONFIG_DIR = "SENTINELINSPECT_CONFIG_DIR"

# sentinelinspect/config/paths.py -> repository root -> configs/
PACKAGED_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def config_dir() -> str:
    """Absolute path to the config directory, overridable by environment.

    Note the limitation this accepts: configs live beside the package rather
    than inside it, so a non-editable wheel install does not carry them. The
    container copies them in and sets the environment variable instead.
    """
    return os.getenv(ENV_CONFIG_DIR) or str(PACKAGED_CONFIG_DIR)
