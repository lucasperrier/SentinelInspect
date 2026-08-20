from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from sentinelinspect.config.load import to_runtime_config
from sentinelinspect.data.datamodule import CrackDataModule
from sentinelinspect.models.factory import build_model

# MLflow is a training-time dependency only, kept out of the core install so the
# inference container does not ship Flask, SQLAlchemy, alembic and friends.
try:
    import mlflow
    from pytorch_lightning.loggers import MLFlowLogger
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise ModuleNotFoundError(
        "Training requires MLflow, which is not part of the core install so that "
        "the inference container stays slim. Install it with:\n"
        '    pip install -e ".[train]"'
    ) from exc


def flatten_dict(cfg: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    runtime = to_runtime_config(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if runtime.trainer and runtime.trainer.deterministic:
        seed_everything(runtime.seed, workers=True)

    tracking_uri = runtime.mlflow.tracking_uri
    experiment_name = runtime.mlflow.experiment_name
    run_name = runtime.run_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_dict(cfg_dict))

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            dirpath=f"runs/{experiment_name}-{run_name}",
            filename=f"{runtime.model.name}" + "-{epoch:02d}-{val_loss:.4f}",
        )

        callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_id=run.info.run_id,
        )

        datamodule = CrackDataModule(
            batch_size=runtime.data.batch_size,
            num_workers=runtime.data.num_workers,
            preprocessing=runtime.preprocessing.model_dump() if runtime.preprocessing else None,
            manifest_path=runtime.data.manifest_path,
            train_split_path=runtime.data.train_split_path,
            val_split_path=runtime.data.val_split_path,
            test_split_path=runtime.data.test_split_path,
            raw_root=runtime.data.raw_root,
            validate_artifacts=runtime.data.validate_artifacts,
            fail_on_validation_error=runtime.data.fail_on_validation_error,
        )

        model = build_model(runtime.model.model_dump())

        trainer = Trainer(
            max_epochs=runtime.trainer.max_epochs,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=runtime.trainer.log_every_n_steps,
            deterministic=runtime.trainer.deterministic,
            accelerator=runtime.trainer.accelerator,
            devices=runtime.trainer.devices,
            precision=runtime.trainer.precision,
            val_check_interval=runtime.trainer.val_check_interval,
        )

        trainer.fit(model, datamodule=datamodule)

        best_ckpt_path = checkpoint_callback.best_model_path
        if best_ckpt_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = Path("trained_models") / experiment_name / timestamp
            export_dir.mkdir(parents=True, exist_ok=True)
            exported_ckpt_path = export_dir / Path(best_ckpt_path).name
            shutil.copy2(best_ckpt_path, exported_ckpt_path)
            mlflow.log_artifact(str(exported_ckpt_path), artifact_path="trained_models")
            mlflow.log_param("exported_checkpoint", str(exported_ckpt_path))
            print(f"Best checkpoint saved to: {exported_ckpt_path}")

        val_metrics = trainer.validate(datamodule=datamodule, ckpt_path=best_ckpt_path or "best")
        test_metrics = trainer.test(datamodule=datamodule, ckpt_path=best_ckpt_path or "best")

        for metric_dict in val_metrics + test_metrics:
            for key, value in metric_dict.items():
                mlflow.log_metric(key, float(value))

        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            mlflow.log_metric(key, float(value))


if __name__ == "__main__":
    main()