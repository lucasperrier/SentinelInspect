from pathlib import Path

import pandas as pd
from PIL import Image

from sentinelinspect.data.build_manifest import build_manifest


def _make_image(path: Path, size=(32, 32), color=(255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


def test_build_manifest_basic(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"

    _make_image(raw_root / "sdnet2018" / "train" / "Positive" / "img1.jpg")
    _make_image(raw_root / "sdnet2018" / "test" / "Negative" / "img2.jpg")

    df = build_manifest(raw_root)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    expected_cols = {
        "path",
        "relative_path",
        "dataset",
        "split",
        "label",
        "width",
        "height",
        "channels",
        "file_size",
        "sha256",
    }
    assert expected_cols.issubset(set(df.columns))

    assert set(df["dataset"]) == {"sdnet2018"}
    assert set(df["split"].dropna()) == {"train", "test"}
    assert set(df["label"].dropna()) == {"crack", "non_crack"}

    assert (df["width"] == 32).all()
    assert (df["height"] == 32).all()
    assert (df["channels"] == 3).all()
    assert df["file_size"].gt(0).all()
    assert df["sha256"].str.len().eq(64).all()


def test_build_manifest_empty(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    df = build_manifest(raw_root)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "file_size" in df.columns
    assert "sha256" in df.columns
