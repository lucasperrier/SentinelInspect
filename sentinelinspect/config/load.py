from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from sentinelinspect.config.schema import RuntimeConfig


def to_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """
    Convert Hydra/OmegaConf config into validated typed RuntimeConfig.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Expected Hydra config to resolve to a dictionary")
    return RuntimeConfig.model_validate(raw)