#!/usr/bin/env python3
"""Audit Nutrition5k metadata and create an exploratory RGB manifest.

This script intentionally uses only the Python standard library. Raw data stay
under data/raw/ (ignored by Git); the generated manifest is small and may be
versioned as debug evidence. These manifests are not the team's canonical split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    calories_kcal: float
    mass_g: float
    fat_g: float
    carb_g: float
    protein_g: float
    official_split: str
    image_path: str
    image_exists: bool


def parse_metadata(paths: list[Path]) -> tuple[dict[str, tuple[float, ...]], dict[str, int]]:
    records: dict[str, tuple[float, ...]] = {}
    duplicate_ids = 0
    invalid_rows = 0

    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) < 6:
                    invalid_rows += 1
                    continue
                dish_id = row[0].strip()
                try:
                    values = tuple(float(value) for value in row[1:6])
                except ValueError:
                    invalid_rows += 1
                    continue
                if not dish_id.startswith("dish_") or not all(
                    math.isfinite(value) and value >= 0 for value in values
                ):
                    invalid_rows += 1
                    continue
                if dish_id in records:
                    duplicate_ids += 1
                    continue
                records[dish_id] = values

    return records, {
        "metadata_rows_valid": len(records),
        "metadata_rows_invalid": invalid_rows,
        "metadata_duplicate_ids": duplicate_ids,
    }


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(
    dataset_root: Path,
    metadata: dict[str, tuple[float, ...]],
    split_name: str,
    dish_ids: list[str],
) -> tuple[list[DishRecord], int]:
    records: list[DishRecord] = []
    missing_metadata = 0
    for dish_id in dish_ids:
        values = metadata.get(dish_id)
        if values is None:
            missing_metadata += 1
            continue
        image_path = Path("imagery") / "realsense_overhead" / dish_id / "rgb.png"
        calories, mass, fat, carb, protein = values
        records.append(
            DishRecord(
                dish_id=dish_id,
                calories_kcal=calories,
                mass_g=mass,
                fat_g=fat,
                carb_g=carb,
                protein_g=protein,
                official_split=split_name,
                image_path=image_path.as_posix(),
                image_exists=(dataset_root / image_path).is_file(),
            )
        )
    return records, missing_metadata


def select_records(records: list[DishRecord], limit: int, seed: int, require_images: bool) -> list[DishRecord]:
    eligible = [record for record in records if record.image_exists or not require_images]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:limit] if limit >= 0 else eligible


def write_manifest(path: Path, records: list[DishRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DishRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--train-limit", type=int, default=20)
    parser.add_argument("--test-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=539)
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Include only rows whose overhead rgb.png exists locally.",
    )
    args = parser.parse_args()

    metadata_paths = [
        args.dataset_root / "metadata" / "dish_metadata_cafe1.csv",
        args.dataset_root / "metadata" / "dish_metadata_cafe2.csv",
    ]
    train_ids_path = args.dataset_root / "dish_ids" / "splits" / "rgb_train_ids.txt"
    test_ids_path = args.dataset_root / "dish_ids" / "splits" / "rgb_test_ids.txt"
    required_paths = metadata_paths + [train_ids_path, test_ids_path]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        parser.error("missing required files: " + ", ".join(missing_paths))

    metadata, audit = parse_metadata(metadata_paths)
    train_ids = read_ids(train_ids_path)
    test_ids = read_ids(test_ids_path)
    overlap = sorted(set(train_ids) & set(test_ids))
    train_records, train_missing = build_records(args.dataset_root, metadata, "train", train_ids)
    test_records, test_missing = build_records(args.dataset_root, metadata, "test", test_ids)
    selected = select_records(train_records, args.train_limit, args.seed, args.require_images)
    selected += select_records(test_records, args.test_limit, args.seed + 1, args.require_images)
    selected.sort(key=lambda record: (record.official_split, record.dish_id))
    write_manifest(args.output, selected)

    summary = {
        **audit,
        "official_train_ids": len(train_ids),
        "official_test_ids": len(test_ids),
        "train_test_overlap": len(overlap),
        "train_ids_missing_metadata": train_missing,
        "test_ids_missing_metadata": test_missing,
        "local_overhead_images_matched": sum(record.image_exists for record in train_records + test_records),
        "manifest_rows": len(selected),
        "manifest_train_rows": sum(record.official_split == "train" for record in selected),
        "manifest_test_rows": sum(record.official_split == "test" for record in selected),
        "seed": args.seed,
        "require_images": args.require_images,
        "input_sha256": {path.name: file_sha256(path) for path in required_paths},
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
