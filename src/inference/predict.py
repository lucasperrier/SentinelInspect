from __future__ import annotations

from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np
from PIL import Image

from src.config.load import to_runtime_config
from src.models.factory import model_class_for
from src.preprocessing.transforms import build_inference_transforms


@hydra.main(version_base=None, config_path="../../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    runtime = to_runtime_config(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if not runtime.checkpoint_path:
        raise ValueError("checkpoint_path is required for inference")

    model_cls = model_class_for(runtime.model.name)
    model = model_cls.load_from_checkpoint(
        runtime.checkpoint_path, config=runtime.model.model_dump()
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and runtime.device != "cpu" else "cpu")
    model.to(device)

    # Example single-image path; override with CLI:
    # python -m src.inference.predict image_path=/abs/path/image.jpg checkpoint_path=/abs/path.ckpt
    image_path = cfg.get("image_path", None)
    if image_path is None:
        raise ValueError("Please pass image_path=... as Hydra override")

    # the SAME pipeline evaluation uses, so serving cannot drift from scoring
    tfm = build_inference_transforms(
        runtime.preprocessing.model_dump() if runtime.preprocessing else None
    )
    image = Image.open(image_path).convert("RGB")
    x = tfm(image=np.array(image))["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    print({
        "image_path": image_path,
        "predicted_class": pred,
        "probabilities": probs.detach().cpu().tolist(),
        "model": runtime.model.name,
        "checkpoint_path": runtime.checkpoint_path,
    })


if __name__ == "__main__":
    main()