import pandas as pd
import pytest

from sentinelinspect.data.splitters import (
    build_splits_from_manifest,
    split_balance,
    validate_split_ratios,
    write_split_files,
)


def _manifest(n: int = 100, prefix: str = "ds") -> pd.DataFrame:
    """A manifest of `n` distinct images: every sha256 is unique."""
    return pd.DataFrame(
        {
            "relative_path": [f"{prefix}/img_{i:04d}.jpg" for i in range(n)],
            "label": ["crack" if i % 2 == 0 else "non_crack" for i in range(n)],
            "sha256": [f"hash_{i:04d}" for i in range(n)],
        }
    )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_build_splits_is_deterministic_for_same_seed():
    df = _manifest()
    out1 = build_splits_from_manifest(df, seed=42)
    out2 = build_splits_from_manifest(df, seed=42)
    assert out1["split"].tolist() == out2["split"].tolist()


def test_build_splits_changes_with_seed():
    df = _manifest()
    out1 = build_splits_from_manifest(df, seed=42)
    out2 = build_splits_from_manifest(df, seed=43)
    assert out1["split"].tolist() != out2["split"].tolist()


def test_split_is_independent_of_row_order():
    """Shuffling the manifest must not move any image between splits."""
    df = _manifest()
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    a = build_splits_from_manifest(df).set_index("sha256")["split"]
    b = build_splits_from_manifest(shuffled).set_index("sha256")["split"]

    assert a.sort_index().equals(b.sort_index())


def test_split_is_stable_when_the_dataset_grows():
    """Adding images must not reassign the images that were already there.

    This is the property that a seeded shuffle would not give us, and it is
    the reason the split is keyed on a hash rather than on an ordering.
    """
    small = _manifest(200)
    large = _manifest(400)

    before = build_splits_from_manifest(small).set_index("sha256")["split"]
    after = build_splits_from_manifest(large).set_index("sha256")["split"]

    assert after.loc[before.index].equals(before)


# --------------------------------------------------------------------------
# the leakage fix
# --------------------------------------------------------------------------

def test_byte_identical_images_land_in_the_same_split():
    """The regression test for the duplicate-dataset leak.

    Two rows describing the same bytes under different paths must never be
    separated, no matter how many copies exist.
    """
    df = pd.DataFrame(
        {
            "relative_path": (
                [f"ccic/img_{i:04d}.jpg" for i in range(200)]
                + [f"sdnet2018/img_{i:04d}.jpg" for i in range(200)]
            ),
            "label": ["crack" if i % 2 == 0 else "non_crack" for i in range(200)] * 2,
            # the second block is a byte-identical copy of the first
            "sha256": [f"hash_{i:04d}" for i in range(200)] * 2,
        }
    )

    out = build_splits_from_manifest(df)

    splits_per_image = out.groupby("sha256")["split"].nunique()
    assert (splits_per_image == 1).all(), (
        f"{int((splits_per_image > 1).sum())} images were split across more than one set"
    )


def test_path_keyed_split_separates_duplicates():
    """The old behaviour, pinned so the regression above cannot be trivially satisfied.

    Keying on the path is what caused the leak. If this ever stops separating
    duplicates the test above proves nothing.
    """
    df = pd.DataFrame(
        {
            "relative_path": (
                [f"ccic/img_{i:04d}.jpg" for i in range(200)]
                + [f"sdnet2018/img_{i:04d}.jpg" for i in range(200)]
            ),
            "label": ["crack"] * 400,
            "sha256": [f"hash_{i:04d}" for i in range(200)] * 2,
        }
    )

    out = build_splits_from_manifest(df, split_key="relative_path")

    splits_per_image = out.groupby("sha256")["split"].nunique()
    assert (splits_per_image > 1).any()


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_validate_split_ratios_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_split_ratios(0.7, 0.2, 0.2)
    with pytest.raises(ValueError):
        validate_split_ratios(0.7, 0.3, 0.0)


def test_build_splits_requires_the_split_key_column():
    df = pd.DataFrame({"relative_path": ["a.jpg"], "label": ["crack"]})
    with pytest.raises(ValueError, match="sha256"):
        build_splits_from_manifest(df)


def test_missing_split_key_error_lists_available_columns():
    """The failure a typo in the key name produces should say what was there."""
    df = pd.DataFrame({"relative_path": ["a.jpg"], "label": ["crack"]})
    with pytest.raises(ValueError, match="relative_path"):
        build_splits_from_manifest(df, split_key="sha_256")


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def test_write_split_files_creates_expected_files(tmp_path):
    df = pd.DataFrame(
        {
            "relative_path": ["a.jpg", "b.jpg", "c.jpg"],
            "label": ["crack", "non_crack", "crack"],
            "sha256": ["h1", "h2", "h3"],
            "split": ["train", "val", "test"],
        }
    )
    write_split_files(df, tmp_path)

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()
    # robustness.csv was a copy of the entire dataset, train rows included
    assert not (tmp_path / "robustness.csv").exists()


def test_split_balance_reports_proportions_that_sum_to_one():
    out = build_splits_from_manifest(_manifest(400))
    balance = split_balance(out)

    assert set(balance.index) == {"train", "val", "test"}
    for _split_name, row in balance.iterrows():
        assert row.sum() == pytest.approx(1.0, abs=1e-6)


def test_split_balance_is_empty_without_a_label_column():
    df = pd.DataFrame({"sha256": ["h1"], "split": ["train"]})
    assert split_balance(df).empty
