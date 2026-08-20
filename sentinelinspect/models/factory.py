from __future__ import annotations

from typing import Any, Dict, Mapping, Type

from sentinelinspect.models.base import CrackClassifier
from sentinelinspect.models.resnet50 import ResNet50Module
from sentinelinspect.models.vit import VisionTransformerModule

MODEL_REGISTRY: Dict[str, Type[CrackClassifier]] = {
    "resnet50": ResNet50Module,
    "vit": VisionTransformerModule,
}


def build_model(model_cfg: Mapping[str, Any]) -> CrackClassifier:
    """Construct a model from its config block.

    Looks the name up exactly rather than by substring. The previous version
    did `if "vit" in name: ... else: ResNet50Module`, so any unrecognised name
    -- including a typo -- silently produced a ResNet.
    """
    name = str(model_cfg.get("name", "")).lower()
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model name {name!r}. Registered models: {sorted(MODEL_REGISTRY)}"
        ) from None
    return model_cls(dict(model_cfg))


def model_class_for(name: str) -> Type[CrackClassifier]:
    """The class behind a name, for `load_from_checkpoint`."""
    key = str(name).lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown model name {key!r}. Registered models: {sorted(MODEL_REGISTRY)}"
        ) from None
