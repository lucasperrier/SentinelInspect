from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.validate_dataset import validate
from src.preprocessing.transforms import (
    build_train_transforms,
    build_eval_transforms,
)

PATH_CANDIDATES = ("path", "image_path", "relative_path")

class CrackDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = [int(x) for x in labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.long)


class CrackDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        preprocessing: dict | None = None,
        seed: int = 42,
        verbose: bool = False,
        manifest_path: str = "data/processed/manifests/manifest.csv",
        train_split_path: str = "data/processed/splits/train.csv",
        val_split_path: str = "data/processed/splits/val.csv",
        test_split_path: str = "data/processed/splits/test.csv",
        raw_root: str = ".",
        validate_artifacts: bool = True,
        fail_on_validation_error: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.preprocessing = preprocessing or {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "hflip_p": 0.5,
            "brightness_contrast_p": 0.3,
            "shift_scale_rotate_p": 0.3,
            "shift_limit": 0.03,
            "scale_limit": 0.05,
            "rotate_limit": 10,
        }

        self.manifest_path = Path(manifest_path)
        self.train_split_path = Path(train_split_path)
        self.val_split_path = Path(val_split_path)
        self.test_split_path = Path(test_split_path)
        self.raw_root = Path(raw_root)

        self.validate_artifacts = validate_artifacts
        self.fail_on_validation_error = fail_on_validation_error
        
        # Transforms
        self.train_transform = build_train_transforms(self.preprocessing)
        self.eval_transform = build_eval_transforms(self.preprocessing)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # No download. Optional: check file existence.
        required = [self.manifest_path, self.train_split_path, self.val_split_path, self.test_split_path]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required processed artifacts: {missing}")

    def setup(self, stage: Optional[str] = None):
        if self.validate_artifacts:
            report = validate(
                manifest_path=self.manifest_path,
                train_path=self.train_split_path,
                val_path=self.val_split_path,
                test_path=self.test_split_path,
                raw_root=self.raw_root,
            )
            if self.verbose:
                print("[CrackDataModule] validation report:", report)
                for warning in report.warnings:
                    print("[CrackDataModule] warning:", warning)
            if report.errors and self.fail_on_validation_error:
                raise ValueError(
                    "Dataset artifact validation failed:\n- " + "\n- ".join(report.errors)
                )

        train_df = pd.read_csv(self.train_split_path)
        val_df = pd.read_csv(self.val_split_path)
        test_df = pd.read_csv(self.test_split_path)

        self.train_dataset = self._df_to_dataset(train_df, split_name="train", transform=self.train_transform)
        self.val_dataset = self._df_to_dataset(val_df, split_name="val", transform=self.eval_transform)
        self.test_dataset = self._df_to_dataset(test_df, split_name="test", transform=self.eval_transform)

    def _resolve_path_column(self, df: pd.DataFrame) -> str:
        for c in PATH_CANDIDATES:
            if c in df.columns:
                return c
        raise ValueError(f"Missing path column. Expected one of {PATH_CANDIDATES}")

    def _df_to_dataset(self, df: pd.DataFrame, split_name: str, transform):
        if "label" not in df.columns:
            raise ValueError(f"[{split_name}] missing required column: label")

        path_col = self._resolve_path_column(df)
        paths: List[str] = []
        labels: List[int] = []

        for _, row in df.iterrows():
            p = Path(str(row[path_col]))
            p = p if p.is_absolute() else (self.raw_root / p)
            paths.append(str(p))
            raw_label = str(row["label"]).strip().lower()
            if raw_label in {"1", "crack", "positive", "pos"}:
                labels.append(1)
            elif raw_label in {"0", "non_crack", "no_crack", "negative", "neg"}:
                labels.append(0)
            else:
                raise ValueError(f"[{split_name}] unsupported label value: {row['label']!r}")

        return CrackDataset(paths, labels, transform=transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
