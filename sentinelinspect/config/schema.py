from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    scheduler: str | None = None


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
    registry_uri: str | None = None
    use_mlflow: bool = True


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class ReviewConfig(BaseModel):
    """Triage policy: the confidence band routed to a human.

    Lives in config rather than in code because the right band depends on how
    much manual review capacity exists, which is an operational decision, not a
    modelling one. Defaults are tuned on the validation split -- see
    `evaluation.metrics.tune_review_band`.
    """

    model_config = ConfigDict(extra="forbid")

    lower: float = Field(default=0.35, ge=0.0, le=1.0)
    upper: float = Field(default=0.65, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_band(self) -> ReviewConfig:
        if self.lower > self.upper:
            raise ValueError(f"review.lower ({self.lower}) must not exceed review.upper ({self.upper})")
        return self


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

    trainer: TrainerConfig | None = None
    service: ServiceConfig | None = None
    preprocessing: PreprocessingConfig | None = None
    review: ReviewConfig | None = None

    checkpoint_path: str | None = None
    output_dir: str | None = None
    split: Literal["train", "val", "test"] | None = None
    device: Literal["cpu", "gpu", "auto"] | None = None
    image_path: str | None = None

    @model_validator(mode="after")
    def validate_task_requirements(self) -> RuntimeConfig:
        if self.task in {"train", "eval"} and self.trainer is None:
            raise ValueError("trainer is required for task=train/eval")
        if self.task == "eval" and self.split is None:
            raise ValueError("split is required for task=eval")
        return self