"""Measure paired brightness and blur sensitivity on one EuroSAT development fold."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline import BINS, CLASSES, FOLDS, NEIGHBORS, development_rows, extract_features, metrics


def select_sample(rows, fold, seed=42, per_class=50):
    """Sample before prediction with the same source images for all conditions."""
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(len(CLASSES)):
        candidates = sorted(
            (row for row in rows if row[1] == label and row[2] == fold),
            key=lambda row: row[0].as_posix(),
        )
        indices = rng.choice(len(candidates), size=min(per_class, len(candidates)), replace=False)
        selected.extend(candidates[index] for index in sorted(indices))
    return selected


def altered_features(data_root, rows, bins, condition, value):
    """Use saved lossless variants so the clean baseline feature code is reused."""
    with TemporaryDirectory(prefix="eurosat-dev-stress-") as temporary:
        variant_root = Path(temporary)
        variant_rows = []
        for index, (path, label, fold) in enumerate(rows):
            with Image.open(data_root / path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            if condition == "brightness":
                altered = pixels * value
            elif condition == "gaussian_blur":
                altered = gaussian_filter(pixels, sigma=(value, value, 0), mode="reflect")
            else:
                raise ValueError(f"Unknown stress condition: {condition}")
            filename = Path(f"{index}.png")
            Image.fromarray(np.clip(np.rint(altered), 0, 255).astype(np.uint8)).save(
                variant_root / filename
            )
            variant_rows.append((filename, label, fold))
        return extract_features(variant_root, variant_rows)[bins]


def compare_to_clean(scores, clean):
    """Report the signed change from the corresponding clean predictions."""
    return {
        "accuracy": scores["accuracy"] - clean["accuracy"],
        "macro_f1": scores["macro_f1"] - clean["macro_f1"],
        "per_class_recall": {
            label: scores["per_class_recall"][label] - clean["per_class_recall"][label]
            for label in CLASSES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dev-stress"))
    parser.add_argument("--fold", type=int, choices=FOLDS, default=0)
    parser.add_argument("--bins", type=int, choices=BINS, default=16)
    parser.add_argument("--k", type=int, choices=NEIGHBORS, default=5)
    args = parser.parse_args()

    rows = development_rows(args.manifest)
    training_rows = [row for row in rows if row[2] != args.fold]
    sample = select_sample(rows, args.fold)
    train_labels = np.asarray([label for _, label, _ in training_rows], dtype=np.int64)
    sample_labels = np.asarray([label for _, label, _ in sample], dtype=np.int64)
    print(f"Training on {len(training_rows)} development images and evaluating "
          f"{len(sample)} paired images from development fold {args.fold}.", flush=True)
    classifier = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=args.k, metric="euclidean", algorithm="brute"),
    )
    with threadpool_limits(limits=2):
        classifier.fit(extract_features(args.data_root, training_rows)[args.bins], train_labels)
        clean_features = extract_features(args.data_root, sample)[args.bins]
        clean = metrics(sample_labels, classifier.predict(clean_features))
        conditions = [{"name": "clean", **clean, "delta_from_clean": compare_to_clean(clean, clean)}]
        for condition, amount in (
            ("brightness", 0.7),
            ("brightness", 1.3),
            ("gaussian_blur", 1),
            ("gaussian_blur", 2),
        ):
            features = altered_features(args.data_root, sample, args.bins, condition, amount)
            scores = metrics(sample_labels, classifier.predict(features))
            parameter = "multiplier" if condition == "brightness" else "sigma_px"
            conditions.append({
                "name": condition,
                parameter: amount,
                **scores,
                "delta_from_clean": compare_to_clean(scores, clean),
            })
            print(f"{condition} {amount}: accuracy={scores['accuracy']:.4f}, "
                  f"macro-F1={scores['macro_f1']:.4f}", flush=True)

    report = {
        "evaluation_set": "development stress pilot",
        "metric_interpretation": (
            "Preliminary paired stress results on one development fold. "
            "No final holdout images were loaded. If the kNN configuration was selected using "
            "the same development folds, these are development tuning estimates."
        ),
        "class_order": CLASSES,
        "confusion_matrix_axes": {"rows": "true label", "columns": "predicted label"},
        "validation_fold": args.fold,
        "training_folds": [fold for fold in FOLDS if fold != args.fold],
        "n_training": len(training_rows),
        "sample_seed": 42,
        "target_per_class": 50,
        "actual_per_class": clean["per_class_support"],
        "sample_paths": [path.as_posix() for path, _, _ in sample],
        "config": {"bins_per_channel": args.bins, "k": args.k, "distance": "euclidean"},
        "conditions": conditions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "stress_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
