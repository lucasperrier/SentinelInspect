from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from sentinelinspect.data.datamodule import CrackDataModule


def _make_image(path: Path, size=(32, 32), color=(255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_minimal_artifacts(tmp_path: Path):
    raw_root = tmp_path / "raw"
    rel1 = Path("ccic/Positive/img1.jpg")
    rel2 = Path("ccic/Negative/img2.jpg")
    rel3 = Path("sdnet2018/Positive/img3.jpg")

    _make_image(raw_root / rel1)
    _make_image(raw_root / rel2)
    _make_image(raw_root / rel3)

    manifest = tmp_path / "manifest.csv"
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"

    _write_csv(
        manifest,
        [
            {"relative_path": str(rel1), "label": 1},
            {"relative_path": str(rel2), "label": 0},
            {"relative_path": str(rel3), "label": 1},
        ],
    )
    _write_csv(train, [{"relative_path": str(rel1), "label": 1}])
    _write_csv(val, [{"relative_path": str(rel2), "label": 0}])
    _write_csv(test, [{"relative_path": str(rel3), "label": 1}])

    return raw_root, manifest, train, val, test


def test_datamodule_loads_from_processed_split_csvs(tmp_path: Path):
    raw_root, manifest, train, val, test = _build_minimal_artifacts(tmp_path)

    dm = CrackDataModule(
        batch_size=2,
        num_workers=0,
        manifest_path=str(manifest),
        train_split_path=str(train),
        val_split_path=str(val),
        test_split_path=str(test),
        raw_root=str(raw_root),
        validate_artifacts=True,
        fail_on_validation_error=True,
    )
    dm.prepare_data()
    dm.setup()

    xb, yb = next(iter(dm.train_dataloader()))
    assert isinstance(xb, torch.Tensor)
    assert isinstance(yb, torch.Tensor)
    assert xb.ndim == 4
    assert xb.shape[1:] == (3, 224, 224)
    assert yb.dtype == torch.long


def test_datamodule_fails_fast_on_overlap_when_validation_enabled(tmp_path: Path):
    raw_root = tmp_path / "raw"
    rel = Path("ccic/Positive/shared.jpg")
    _make_image(raw_root / rel)

    manifest = tmp_path / "manifest.csv"
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"

    _write_csv(manifest, [{"relative_path": str(rel), "label": 1}])
    _write_csv(train, [{"relative_path": str(rel), "label": 1}])
    _write_csv(val, [{"relative_path": str(rel), "label": 1}])   # overlap with train
    pd.DataFrame(columns=["relative_path", "label"]).to_csv(test, index=False)

    dm = CrackDataModule(
        batch_size=2,
        num_workers=0,
        manifest_path=str(manifest),
        train_split_path=str(train),
        val_split_path=str(val),
        test_split_path=str(test),
        raw_root=str(raw_root),
        validate_artifacts=True,
        fail_on_validation_error=True,
    )

    with pytest.raises(ValueError, match="validation failed"):
        dm.setup()


def test_datamodule_can_continue_when_validation_errors_allowed(tmp_path: Path):
    raw_root = tmp_path / "raw"
    rel = Path("ccic/Positive/missing.jpg")  # not created

    manifest = tmp_path / "manifest.csv"
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"

    _write_csv(manifest, [{"relative_path": str(rel), "label": 1}])
    _write_csv(train, [{"relative_path": str(rel), "label": 1}])
    pd.DataFrame(columns=["relative_path", "label"]).to_csv(val, index=False)
    pd.DataFrame(columns=["relative_path", "label"]).to_csv(test, index=False)

    dm = CrackDataModule(
        batch_size=1,
        num_workers=0,
        manifest_path=str(manifest),
        train_split_path=str(train),
        val_split_path=str(val),
        test_split_path=str(test),
        raw_root=str(raw_root),
        validate_artifacts=True,
        fail_on_validation_error=False,  # allow warnings
    )

    dm.setup()
    assert dm.train_dataset is not None


def test_datamodule_rejects_non_binary_labels(tmp_path: Path):
    raw_root = tmp_path / "raw"
    rel = Path("ccic/Positive/img.jpg")
    _make_image(raw_root / rel)

    manifest = tmp_path / "manifest.csv"
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"

    _write_csv(manifest, [{"relative_path": str(rel), "label": 2}])  # invalid for binary contract
    _write_csv(train, [{"relative_path": str(rel), "label": 2}])
    pd.DataFrame(columns=["relative_path", "label"]).to_csv(val, index=False)
    pd.DataFrame(columns=["relative_path", "label"]).to_csv(test, index=False)

    dm = CrackDataModule(
        batch_size=1,
        num_workers=0,
        manifest_path=str(manifest),
        train_split_path=str(train),
        val_split_path=str(val),
        test_split_path=str(test),
        raw_root=str(raw_root),
        validate_artifacts=False,  # isolate label check in datamodule
    )

    # the mapping in _df_to_dataset rejects anything outside the known label
    # vocabulary, so an out-of-contract value never reaches a numeric check
    with pytest.raises(ValueError, match="unsupported label value"):
        dm.setup()