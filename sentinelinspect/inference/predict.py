"""Single-image prediction CLI.

A thin adapter over `Predictor`: parse config, build the core, print the
contract. Any logic that appears here rather than in the Predictor is logic the
HTTP route will not share.

    python -m sentinelinspect.inference.predict \
        "checkpoint_path='runs/<exp>/<model>-epoch=00-val_loss=0.20.ckpt'" \
        "image_path='data/raw/ccic/Positive/00001.jpg'"

Quote the overrides: checkpoint filenames contain '=', which Hydra's override
parser reads as a separator.
"""

from __future__ import annotations

import json
import sys

import hydra
from omegaconf import DictConfig

from sentinelinspect.config.load import to_runtime_config
from sentinelinspect.config.paths import config_dir
from sentinelinspect.inference.model_loader import CheckpointNotFoundError
from sentinelinspect.inference.predictor import InvalidImageError, Predictor


@hydra.main(version_base=None, config_path=config_dir(), config_name="inference")
def main(cfg: DictConfig) -> None:
    runtime = to_runtime_config(cfg)

    if not runtime.image_path:
        raise SystemExit("image_path is required, e.g. \"image_path='path/to/image.jpg'\"")

    try:
        predictor = Predictor.from_runtime_config(runtime)
        prediction = predictor.predict_image(runtime.image_path)
    except CheckpointNotFoundError as exc:
        raise SystemExit(f"error: {exc}") from exc
    except InvalidImageError as exc:
        raise SystemExit(f"error: {exc}") from exc

    json.dump(prediction.model_dump(mode="json"), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
