from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Tuple

import pandas as pd

DEFAULT_SPLIT_KEY = "sha256"

DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)


def stable_hash_to_unit_interval(value: str, seed: int = 42) -> float:
    payload = f"{seed}::{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    int_value = int(digest[:16], 16)
    max_value = float(16**16 - 1)
    return int_value / max_value


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) <=0:
        raise ValueError("All split ratios must be > 0.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total-1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")
    

def assign_split(
    key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> str:
    score = stable_hash_to_unit_interval(key, seed=seed)
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"

def build_splits_from_manifest(
        manifest_df: pd.DataFrame,
        train_ratio: float = DEFAULT_SPLIT_RATIOS[0],
        val_ratio: float = DEFAULT_SPLIT_RATIOS[1],
        test_ratio: float = DEFAULT_SPLIT_RATIOS[2],
        seed: int = 42,
        split_key: str = DEFAULT_SPLIT_KEY,
) -> pd.DataFrame:
    if split_key not in manifest_df.columns:
        raise ValueError(
            f"Manifest is missing the split key column: {split_key!r}. "
            f"Available columns: {sorted(manifest_df.columns)}"
        )

    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    df = manifest_df.copy()
    df["split"] = df[split_key].astype(str).map(
        lambda key: assign_split(key, train_ratio, val_ratio, seed=seed)
    )

    sort_by = ["split", "relative_path"] if "relative_path" in df.columns else ["split", split_key]
    return df.sort_values(sort_by).reset_index(drop=True)


def split_balance(df: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    if label_column not in df.columns or "split" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("split")[label_column]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .round(4)
    )


def write_split_files(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name].copy()
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Manifest format not supported. Use .csv or .parquet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic train/val/test split files.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/processed/manifests/manifest.csv"),
        help="Path to the input manifest (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Output directory for train.csv, val.csv, test.csv.",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0], help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1], help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2], help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used in deterministic hash splitting.")
    parser.add_argument(
        "--split-key",
        type=str,
        default=DEFAULT_SPLIT_KEY,
        help="Manifest column the split is keyed on. Use a content hash to prevent duplicate leakage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_df = load_manifest(args.manifest_path)
    split_df = build_splits_from_manifest(
        manifest_df=manifest_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_key=args.split_key,
    )
    write_split_files(split_df, args.output_dir)

    counts = split_df["split"].value_counts().to_dict()
    print(f"Loaded manifest rows: {len(manifest_df)}")
    print(f"Split key: {args.split_key}")
    print(f"Split counts: {counts}")

    balance = split_balance(split_df)
    if not balance.empty:
        print("\nClass balance per split:")
        print(balance.to_string())

    print(f"\nWrote split files to: {args.output_dir}")


if __name__ == "__main__":
    main()