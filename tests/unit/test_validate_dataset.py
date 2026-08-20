from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.validate_dataset import validate


def _make_image(path: Path, size=(8, 8), color=(255, 255, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------
# the happy path and the filesystem checks
# --------------------------------------------------------------------------

def test_validate_passes_with_valid_manifest_and_splits(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    img1_rel = Path("ccic/Positive/img1.jpg")
    img2_rel = Path("ccic/Negative/img2.jpg")
    _make_image(raw_root / img1_rel)
    _make_image(raw_root / img2_rel, color=(0, 0, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_csv(
        manifest_path,
        [
            {"relative_path": str(img1_rel), "label": "crack", "sha256": "h1"},
            {"relative_path": str(img2_rel), "label": "non_crack", "sha256": "h2"},
        ],
    )

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    _write_csv(train_path, [{"relative_path": str(img1_rel), "label": "crack", "sha256": "h1"}])
    _write_csv(val_path, [{"relative_path": str(img2_rel), "label": "non_crack", "sha256": "h2"}])
    pd.DataFrame(columns=["relative_path", "label", "sha256"]).to_csv(test_path, index=False)

    report = validate(
        manifest_path=manifest_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        raw_root=raw_root,
    )

    assert report.ok
    assert report.errors == []
    assert report.warnings == []
    assert report.total_rows == 2
    assert report.missing_files == 0
    assert report.corrupt_images == 0
    assert report.unreadable_files == 0


def test_validate_fails_on_missing_files(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest_path = tmp_path / "manifest.csv"
    _write_csv(manifest_path, [{"relative_path": "does/not/exist.jpg", "label": "crack", "sha256": "h1"}])

    report = validate(manifest_path=manifest_path, raw_root=raw_root)

    assert not report.ok
    assert report.missing_files == 1
    assert any("missing files: 1" in e for e in report.errors)


def test_validate_detects_corrupt_image(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    rel = Path("ccic/Positive/corrupt.jpg")
    bad_path = raw_root / rel
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"not-an-image")

    manifest_path = tmp_path / "manifest.csv"
    _write_csv(manifest_path, [{"relative_path": str(rel), "label": "crack", "sha256": "h1"}])

    report = validate(manifest_path=manifest_path, raw_root=raw_root)

    assert report.corrupt_images == 1
    assert any("corrupt/unidentified images: 1" in e for e in report.errors)


def test_skip_file_checks_does_not_touch_the_filesystem(tmp_path: Path):
    """Skipping the walk must not silently invent a pass.

    The tabular checks still run; nothing is reported about files that were
    never looked at. Tests rely on this to avoid writing real images to disk.
    """
    manifest_path = tmp_path / "manifest.csv"
    _write_csv(manifest_path, [{"relative_path": "does/not/exist.jpg", "label": "crack", "sha256": "h1"}])

    report = validate(manifest_path=manifest_path, raw_root=tmp_path, check_files=False)

    assert report.missing_files == 0
    assert report.ok


# --------------------------------------------------------------------------
# leakage: the reason this module exists
# --------------------------------------------------------------------------

def test_validate_detects_path_overlap(tmp_path: Path):
    """The same file listed in two splits -- a build bug in the splitter."""
    manifest_path = tmp_path / "manifest.csv"
    rel = "ccic/Positive/shared.jpg"
    _write_csv(manifest_path, [{"relative_path": rel, "label": "crack", "sha256": "h1"}])

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    _write_csv(train_path, [{"relative_path": rel, "label": "crack", "sha256": "h1"}])
    _write_csv(val_path, [{"relative_path": rel, "label": "crack", "sha256": "h1"}])

    report = validate(
        manifest_path=manifest_path,
        train_path=train_path,
        val_path=val_path,
        check_files=False,
    )

    assert report.split_overlap_paths.get("train-val") == 1
    assert any("path overlap detected between train and val" in e for e in report.errors)


def test_validate_detects_content_overlap_the_path_check_cannot_see(tmp_path: Path):
    """The bug this whole change exists to catch.

    Two different filenames, identical bytes, one in train and one in test.
    The path check sees two distinct strings and passes. Only the content
    check notices that the test set was already trained on.
    """
    manifest_path = tmp_path / "manifest.csv"
    _write_csv(
        manifest_path,
        [
            {"relative_path": "ccic/Positive/img.jpg", "label": "crack", "sha256": "SAME"},
            {"relative_path": "sdnet2018/Positive/img.jpg", "label": "crack", "sha256": "SAME"},
            {"relative_path": "ccic/Negative/other.jpg", "label": "non_crack", "sha256": "OTHER"},
        ],
    )

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_csv(train_path, [{"relative_path": "ccic/Positive/img.jpg", "label": "crack", "sha256": "SAME"}])
    _write_csv(test_path, [{"relative_path": "sdnet2018/Positive/img.jpg", "label": "crack", "sha256": "SAME"}])

    report = validate(
        manifest_path=manifest_path,
        train_path=train_path,
        test_path=test_path,
        check_files=False,
    )

    # the check that used to be the only one: sees nothing
    assert report.split_overlap_paths.get("train-test") == 0
    # the check that was added: sees the leak
    assert report.split_overlap_content.get("train-test") == 1
    assert not report.ok
    assert any("content overlap detected between train and test" in e for e in report.errors)


def test_missing_sha256_column_warns_that_leakage_cannot_be_detected(tmp_path: Path):
    """A manifest with no content hash gets zero leakage checking.

    Without this warning it would report PASSED while checking nothing, which
    is the same failure mode the content check was added to fix.
    """
    manifest_path = tmp_path / "manifest.csv"
    _write_csv(manifest_path, [{"relative_path": "a.jpg", "label": "crack"}])

    report = validate(manifest_path=manifest_path, check_files=False)

    assert report.ok
    assert any("cannot be detected" in w for w in report.warnings)


# --------------------------------------------------------------------------
# severity: warnings are not errors
# --------------------------------------------------------------------------

def test_content_duplicates_inside_one_split_warn_but_do_not_fail(tmp_path: Path):
    """Duplication within a split over-weights an image; it does not
    invalidate a measurement. It must not halt the pipeline."""
    manifest_path = tmp_path / "manifest.csv"
    _write_csv(
        manifest_path,
        [
            {"relative_path": "ccic/a.jpg", "label": "crack", "sha256": "SAME"},
            {"relative_path": "sdnet2018/a.jpg", "label": "crack", "sha256": "SAME"},
        ],
    )
    train_path = tmp_path / "train.csv"
    _write_csv(
        train_path,
        [
            {"relative_path": "ccic/a.jpg", "label": "crack", "sha256": "SAME"},
            {"relative_path": "sdnet2018/a.jpg", "label": "crack", "sha256": "SAME"},
        ],
    )

    report = validate(manifest_path=manifest_path, train_path=train_path, check_files=False)

    assert report.ok
    assert report.errors == []
    assert report.duplicate_content == 1
    assert any("byte-identical" in w for w in report.warnings)


def test_label_conflicts_are_errors(tmp_path: Path):
    """Identical bytes carrying disagreeing labels is contradictory training
    data. Content-keyed splitting hides it, so the validator must not."""
    manifest_path = tmp_path / "manifest.csv"
    _write_csv(
        manifest_path,
        [
            {"relative_path": "ccic/Positive/a.jpg", "label": "crack", "sha256": "SAME"},
            {"relative_path": "ccic/Negative/a.jpg", "label": "non_crack", "sha256": "SAME"},
        ],
    )

    report = validate(manifest_path=manifest_path, check_files=False)

    assert not report.ok
    assert report.label_conflicts == 1
    assert any("more than one label" in e for e in report.errors)
