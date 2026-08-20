from __future__ import annotations


import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "val", "test")

@dataclass
class ManifestRecord:
    path: str
    relative_path: str
    dataset: str
    split: str | None
    label: str | None
    width: int | None
    height: int | None
    channels: int | None
    file_size: int
    sha256: str


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_split(parts: tuple[str, ...]) -> str | None:
    lower_parts = [part.lower() for part in parts]
    for split in DEFAULT_SPLITS:
        if split in lower_parts: 
            return split
    return None


def infer_label(parts: tuple[str, ...]) -> str | None:
    lower_parts = [part.lower() for part in parts]

    positive_tokens = {"crack", "cracked", "positive", "pos", "1"}
    negative_tokens = {"non_crack", "non-crack", "noncrack", "negative", "neg", "0", "no_crack"}

    for part in lower_parts:
        if part in positive_tokens:
            return "crack"
        if part in negative_tokens:
            return "non_crack"
        
    for part in lower_parts:
        if "non" in part and "crack" in part:
            return "non_crack"
        if "crack" in part:
            return "crack"
        
    return None


def get_image_metadata(path: Path) -> tuple[int | None, int | None, int | None]:
    try:
        with Image.open(path) as img:
            width, height, = img.size
            bands = img.getbands()
            channels = len(bands) if bands is not None else None
            return width, height, channels
    except (UnidentifiedImageError, OSError):
        return None, None, None
    

def build_record(path: Path, raw_root: Path) -> ManifestRecord:
    relative_path = path.relative_to(raw_root)
    parts = relative_path.parts

    dataset = parts[0] if len(parts) > 0 else "unknown"
    split = infer_split(parts)
    label = infer_label(parts)
    width, height, channels = get_image_metadata(path)
    file_size = path.stat().st_size
    sha256 = sha256_file(path)

    return ManifestRecord(
        path=str(path.resolve()),
        relative_path=str(relative_path),
        dataset=dataset,
        split=split,
        label=label,
        width=width,
        height=height,
        channels=channels,
        file_size=file_size,
        sha256=sha256,
    )


def build_manifest(raw_root: Path) -> pd.DataFrame:
    records = [asdict(build_record(path, raw_root)) for path in iter_image_files(raw_root)]
    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    return df.sort_values(["dataset", "split", "label", "relative_path"], na_position="last").reset_index(drop=True)


def save_manifest(df: pd.DataFrame, output_csv: Path, output_json: Path | None = None) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest for image datasets in data/raw.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing raw datasets.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/manifests/manifest.csv"),
        help="Path to the output CSV manifest.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/manifests/manifest.json"),
        help="Optional path to the output JSON manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.raw_root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {args.raw_root}")

    df = build_manifest(args.raw_root)
    save_manifest(df, args.output_csv, args.output_json)

    print(f"Scanned {len(df)} image files from {args.raw_root}")
    print(f"CSV manifest written to: {args.output_csv}")
    if args.output_json is not None:
        print(f"JSON manifest written to: {args.output_json}")


if __name__ == "__main__":
    main()