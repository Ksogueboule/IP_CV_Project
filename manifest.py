"""Build the reproducible EuroSAT split shared by both classifiers."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold


CLASSES = ("Forest", "HerbaceousVegetation", "River", "SeaLake")
EXPECTED_IMAGES = 11_500
TEST_FRACTION = 0.15
N_FOLDS = 3


def inventory(data_root):
    rows_by_class = {label: [] for label in CLASSES}
    source_counts = Counter()
    duplicate_counts = Counter()
    seen = {}
    for label in CLASSES:
        directory = data_root / label
        if not directory.is_dir():
            raise ValueError(f"Missing class folder: {directory}")
        files = sorted(path for path in directory.rglob("*")
                       if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})
        if not files:
            raise ValueError(f"No JPEG files in {directory}")
        for path in files:
            relative = path.relative_to(data_root).as_posix()
            try:
                with Image.open(path) as image:
                    if image.format != "JPEG" or image.mode != "RGB" or image.size != (64, 64):
                        raise ValueError(f"Expected a 64x64 RGB JPEG: {relative}")
                    group_id = hashlib.sha256(image.tobytes()).hexdigest()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Could not decode image: {relative}") from exc
            source_counts[label] += 1
            previous = seen.get(group_id)
            if previous:
                if previous["label"] != label:
                    raise ValueError(f"Identical pixels with conflicting labels: "
                                     f"{previous['path']} and {relative}")
                duplicate_counts[label] += 1
                continue
            row = {"path": relative, "label": label, "group_id": group_id}
            seen[group_id] = row
            rows_by_class[label].append(row)
    return rows_by_class, source_counts, duplicate_counts


def partition(rows_by_class, seed):
    dev, test = [], []
    for label, rows in rows_by_class.items():
        shuffled = rows.copy()
        random.Random(f"{seed}:{label}").shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * TEST_FRACTION))
        if len(shuffled) - test_count < N_FOLDS:
            raise ValueError(f"Not enough unique {label} images for holdout and three folds")
        test.extend({**row, "split": "test", "fold": ""} for row in shuffled[:test_count])
        dev.extend({**row, "split": "development", "fold": ""} for row in shuffled[test_count:])

    labels = [row["label"] for row in dev]
    groups = [row["group_id"] for row in dev]
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(dev, labels, groups)):
        for index in validation:
            dev[index]["fold"] = fold
    rows = sorted([*dev, *test], key=lambda row: row["path"])
    if len({row["group_id"] for row in rows}) != len(rows):
        raise RuntimeError("Split validation failed: duplicate group assigned more than once")
    for fold in range(N_FOLDS):
        found = {row["label"] for row in dev if row["fold"] == fold}
        if found != set(CLASSES):
            raise ValueError(f"Development fold {fold} lacks one or more classes")
    return rows


def build(data_root, output_dir, seed):
    rows_by_class, source_counts, duplicates = inventory(data_root)
    rows = partition(rows_by_class, seed)
    split_counts = {
        split: Counter(row["label"] for row in rows if row["split"] == split)
        for split in ("development", "test")
    }
    source_total = sum(source_counts.values())
    report = {
        "seed": seed,
        "source_total": source_total,
        "expected_source_total": EXPECTED_IMAGES,
        "source_matches_expected": source_total == EXPECTED_IMAGES,
        "exact_duplicates_removed": sum(duplicates.values()),
        "unique_total": len(rows),
        "development_total": sum(split_counts["development"].values()),
        "test_total": sum(split_counts["test"].values()),
        "class_counts": {
            label: {
                "source": source_counts[label],
                "exact_duplicates_removed": duplicates[label],
                "development": split_counts["development"][label],
                "test": split_counts["test"][label],
                "test_fraction": split_counts["test"][label] / len(rows_by_class[label]),
            }
            for label in CLASSES
        },
        "development_fold_counts": {
            fold: dict(Counter(
                row["label"] for row in rows
                if row["split"] == "development" and row["fold"] == fold
            ))
            for fold in range(N_FOLDS)
        },
        "limitations": [
            "Exact duplicates were identified from decoded RGB pixels. Near duplicates were not scanned.",
            "Grouping uses exact pixel hashes. No verified geolocation metadata was supplied; spatial separation is unverified.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "group_id", "split", "fold"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing the four EuroSAT class folders")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/manifest"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = build(args.data_root, args.output_dir, args.seed)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
