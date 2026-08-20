from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_data_path: str
    interim_data_path: str
    processed_data_path: str

    manifest_path: str
    train_split_path: str
    val_split_path: str
    test_split_path: str
    raw_root: str = "."
    validate_artifacts: bool = True
    fail_on_validation_error: bool = True

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    image_size: int = 224


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    model: str
    num_classes: int = Field(gt=1)
    pretrained: bool = True
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    optimizer: str
    scheduler: Optional[str] = None


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_epochs: int = Field(gt=0)
    accelerator: Literal["cpu", "gpu", "auto"] = "auto"
    devices: int = Field(gt=0)
    precision: str = "32"
    deterministic: bool = True
    val_check_interval: float = Field(gt=0.0)
    log_every_n_steps: int = Field(gt=0)


class MlflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str
    experiment_name: str
    registry_uri: Optional[str] = None
    use_mlflow: bool = True


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_size: int = Field(gt=0)
    mean: list[float]
    std: list[float]


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: Literal["train", "eval", "inference"]
    seed: int = 42
    run_name: str

    data: DataConfig
    model: ModelConfig
    mlflow: MlflowConfig

    trainer: Optional[TrainerConfig] = None
    service: Optional[ServiceConfig] = None
    preprocessing: Optional[PreprocessingConfig] = None

    checkpoint_path: Optional[str] = None
    split: Optional[Literal["train", "val", "test"]] = None
    device: Optional[Literal["cpu", "gpu", "auto"]] = None
    image_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_task_requirements(self) -> "RuntimeConfig":
        if self.task in {"train", "eval"} and self.trainer is None:
            raise ValueError("trainer is required for task=train/eval")
        if self.task == "eval" and self.split is None:
            raise ValueError("split is required for task=eval")
        return self