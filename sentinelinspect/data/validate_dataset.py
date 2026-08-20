from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

# Ordered by preference. `relative_path` first: it is portable across machines
# and directory names, where an absolute "path" column is neither.
PATH_CANDIDATES = ("relative_path", "image_path", "path")
CONTENT_COLUMN = "sha256"
LABEL_COLUMN = "label"
REQUIRED_BASE_COLUMNS = {LABEL_COLUMN}


@dataclass
class ValidationReport:
    total_rows: int
    duplicate_rows: int
    duplicate_paths: int
    duplicate_content: int
    unreadable_files: int
    corrupt_images: int
    missing_files: int
    label_conflicts: int
    class_counts: dict
    split_overlap_paths: dict
    split_overlap_content: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def resolve_path_column(df: pd.DataFrame) -> str | None:
    for c in PATH_CANDIDATES:
        if c in df.columns:
            return c
    return None


def resolve_content_column(df: pd.DataFrame) -> str | None:
    return CONTENT_COLUMN if CONTENT_COLUMN in df.columns else None


def check_required_columns(df: pd.DataFrame, name: str) -> tuple[list[str], str | None]:
    errors: list[str] = []
    path_col = resolve_path_column(df)

    if path_col is None:
        errors.append(f"[{name}] missing path column. Expected one of {PATH_CANDIDATES}")
    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"[{name}] missing required columns: {sorted(missing)}")

    return errors, path_col


def check_duplicates(df: pd.DataFrame, name: str, path_col: str | None) -> tuple[list[str], int, int]:
    errors = []
    dup_rows = int(df.duplicated().sum())
    dup_paths = int(df.duplicated(subset=[path_col]).sum()) if path_col else 0

    if dup_rows > 0:
        errors.append(f"[{name}] duplicated rows: {dup_rows}")
    if dup_paths > 0:
        errors.append(f"[{name}] duplicated {path_col} values: {dup_paths}")

    return errors, dup_rows, dup_paths


def check_content_duplicates(df: pd.DataFrame, name: str, content_col: str | None) -> tuple[list[str], int]:
    if content_col is None or content_col not in df.columns:
        return [], 0

    dup_content = int(df.duplicated(subset=[content_col]).sum())
    if dup_content == 0:
        return [], 0

    unique = int(df[content_col].nunique())
    return (
        [f"[{name}] {dup_content} rows are byte-identical copies of another row "
         f"({unique} unique images across {len(df)} rows)"],
        dup_content,
    )


def check_label_conflicts(
    df: pd.DataFrame,
    name: str,
    content_col: str | None,
    label_col: str = LABEL_COLUMN,
) -> tuple[list[str], int]:
    """The same image bytes carrying more than one label.

    Content-keyed splitting keeps these rows together, so this is never
    leakage -- it is contradictory training data, which is worse and quieter.
    """
    if content_col is None or content_col not in df.columns or label_col not in df.columns:
        return [], 0

    labels_per_image = df.groupby(content_col, dropna=False)[label_col].nunique(dropna=False)
    conflicted = labels_per_image[labels_per_image > 1]
    if conflicted.empty:
        return [], 0

    return (
        [f"[{name}] {len(conflicted)} images carry more than one label "
         f"(same bytes, disagreeing {label_col})"],
        int(len(conflicted)),
    )


def check_files_and_images(paths: Iterable[Path]) -> tuple[list[str], int, int, int]:
    errors = []
    missing_files = 0
    unreadable = 0
    corrupt = 0

    for p in paths:
        if not p.exists():
            missing_files += 1
            continue
        if not p.is_file():
            unreadable += 1
            continue
        try:
            with Image.open(p) as img:
                img.verify()
        except UnidentifiedImageError:
            corrupt += 1
        except Exception:
            unreadable += 1

    if missing_files:
        errors.append(f"missing files: {missing_files}")
    if unreadable:
        errors.append(f"unreadable files: {unreadable}")
    if corrupt:
        errors.append(f"corrupt/unidentified images: {corrupt}")

    return errors, missing_files, unreadable, corrupt



def check_split_overlap(
        split_to_df: dict[str, pd.DataFrame],
        split_to_col: dict[str,str],
        kind: str,
) -> tuple[list[str], dict]:
    errors = []
    overlaps = {}
    names = list(split_to_df.keys())

    for i in range(len(names)):
        for j in range (i + 1, len(names)):
            a, b = names[i], names[j]
            col_a, col_b = split_to_col.get(a), split_to_col.get(b)
            if col_a is None or col_b is None:
                continue
            set_a = set(split_to_df[a][col_a].astype(str))
            set_b = set(split_to_df[b][col_b].astype(str))
            inter = set_a.intersection(set_b)
            overlaps[f"{a}-{b}"] = len(inter)
            if inter:
                errors.append(
                    f"{kind} overlap detected between {a} and {b}: {len(inter)} shared images"
                )
    return errors, overlaps



def class_balance(df: pd.DataFrame) -> dict:
    if LABEL_COLUMN not in df.columns:
        return {}
    counts = df[LABEL_COLUMN].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    return pd.read_csv(path)


def validate(
    manifest_path: Path,
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    raw_root: Path | None = None,
    check_files: bool = True,
) -> ValidationReport:
    """Validate a manifest and its split files.

    Set `check_files=False` to skip the filesystem walk. The walk opens every
    image to verify its header, which costs about 2s for 40k files -- cheap
    enough to leave on at training start. The flag exists so unit tests can
    exercise the tabular checks without materialising image files on disk.
    """
    errors: list[str] = []
    warnings: list[str] = []

    manifest = load_csv(manifest_path, "manifest")
    req_errs, manifest_path_col = check_required_columns(manifest, "manifest")
    errors += req_errs

    manifest_content_col = resolve_content_column(manifest)
    if manifest_content_col is None:
        warnings.append(
            f"[manifest] no {CONTENT_COLUMN!r} column: content-level duplication "
            "and leakage cannot be detected"
        )

    dup_errs, dup_rows, dup_paths = check_duplicates(manifest, "manifest", manifest_path_col)
    errors += dup_errs

    content_warns, dup_content = check_content_duplicates(manifest, "manifest", manifest_content_col)
    warnings += content_warns

    conflict_errs, label_conflicts = check_label_conflicts(manifest, "manifest", manifest_content_col)
    errors += conflict_errs

    missing = unreadable = corrupt = 0
    if manifest_path_col is not None and check_files:
        raw_root = raw_root or Path(".")
        file_paths = []
        for p in manifest[manifest_path_col].astype(str).tolist():
            pp = Path(p)
            file_paths.append(pp if pp.is_absolute() else (raw_root / pp))
        file_errs, missing, unreadable, corrupt = check_files_and_images(file_paths)
        errors += file_errs

    splits: dict[str, pd.DataFrame] = {}
    split_path_cols: dict[str, str] = {}
    split_content_cols: dict[str, str] = {}

    for name, p in {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }.items():
        if p is not None and p.exists():
            df = load_csv(p, name)
            s_errs, s_path_col = check_required_columns(df, name)
            errors += s_errs
            d_errs, _, _ = check_duplicates(df, name, s_path_col)
            errors += d_errs

            s_content_col = resolve_content_column(df)
            c_warns, _ = check_content_duplicates(df, name, s_content_col)
            warnings += c_warns

            if s_path_col is not None:
                splits[name] = df
                split_path_cols[name] = s_path_col
            if s_content_col is not None:
                split_content_cols[name] = s_content_col

    overlap_paths: dict = {}
    overlap_content: dict = {}
    if len(splits) >= 2:
        path_errs, overlap_paths = check_split_overlap(splits, split_path_cols, kind="path")
        errors += path_errs

        content_errs, overlap_content = check_split_overlap(splits, split_content_cols, kind="content")
        errors += content_errs

    return ValidationReport(
        total_rows=len(manifest),
        duplicate_rows=dup_rows,
        duplicate_paths=dup_paths,
        duplicate_content=dup_content,
        unreadable_files=unreadable,
        corrupt_images=corrupt,
        missing_files=missing,
        label_conflicts=label_conflicts,
        class_counts=class_balance(manifest),
        split_overlap_paths=overlap_paths,
        split_overlap_content=overlap_content,
        errors=errors,
        warnings=warnings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset integrity and splits.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.csv")
    parser.add_argument("--train", type=Path, default=None, help="Path to train.csv")
    parser.add_argument("--val", type=Path, default=None, help="Path to val.csv")
    parser.add_argument("--test", type=Path, default=None, help="Path to test.csv")
    parser.add_argument("--raw-root", type=Path, default=Path("."), help="Base dir for relative paths")
    parser.add_argument(
        "--skip-file-checks",
        action="store_true",
        help="Skip opening every image; only check the tabular artifacts.",
    )
    args = parser.parse_args()

    report = validate(
        manifest_path=args.manifest,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        raw_root=args.raw_root,
        check_files=not args.skip_file_checks,
    )

    print("=== Dataset Validation Report ===")
    print(f"total_rows:          {report.total_rows}")
    print(f"duplicate_rows:      {report.duplicate_rows}")
    print(f"duplicate_paths:     {report.duplicate_paths}")
    print(f"duplicate_content:   {report.duplicate_content}")
    print(f"label_conflicts:     {report.label_conflicts}")
    print(f"missing_files:       {report.missing_files}")
    print(f"unreadable_files:    {report.unreadable_files}")
    print(f"corrupt_images:      {report.corrupt_images}")
    print(f"class_counts:        {report.class_counts}")
    if report.split_overlap_paths:
        print(f"split_overlap_paths:   {report.split_overlap_paths}")
    if report.split_overlap_content:
        print(f"split_overlap_content: {report.split_overlap_content}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"- {w}")

    if report.errors:
        print("\nValidation FAILED:")
        for e in report.errors:
            print(f"- {e}")
        return 1

    print("\nValidation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())