"""Evaluate the PF2 color histogram and Sobel kNN baseline on development folds."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CLASSES = ("Forest", "HerbaceousVegetation", "River", "SeaLake")
FOLDS = (0, 1, 2)
BINS = (16, 32)
NEIGHBORS = (3, 5, 9)


def development_rows(manifest):
    """Read the saved splits and verify that groups stay within a partition."""
    groups = {}
    paths = set()
    development = []
    with manifest.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"path", "label", "group_id", "split", "fold"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest must contain path, label, group_id, split, and fold.")
        for line, row in enumerate(reader, start=2):
            if not all(row[key] for key in required - {"fold"}):
                raise ValueError(f"Missing manifest field at line {line}.")
            path = Path(row["path"])
            if path.is_absolute() or path.drive or ".." in path.parts:
                raise ValueError(f"Expected a path relative to the data root at line {line}.")
            path_key = path.as_posix().casefold()
            if path_key in paths:
                raise ValueError(f"Repeated image path at line {line}: {path}")
            paths.add(path_key)
            label = row["label"]
            if label not in CLASSES:
                raise ValueError(f"Unexpected class at line {line}: {label}")
            split = row["split"]
            if split == "development":
                try:
                    fold = int(row["fold"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Missing development fold at line {line}.") from exc
                if fold not in FOLDS:
                    raise ValueError(f"Invalid development fold at line {line}: {fold}")
                development.append((path, CLASSES.index(label), fold))
                partition = (split, fold)
            elif split == "test":
                partition = (split, None)
            else:
                raise ValueError(f"Unexpected split at line {line}: {split}")
            group = row["group_id"]
            if groups.setdefault(group, partition) != partition:
                raise ValueError(f"Group crosses partitions or development folds: {group}")
    for fold in FOLDS:
        labels = {label for _, label, assigned in development if assigned == fold}
        if labels != set(range(len(CLASSES))):
            raise ValueError(f"Development fold {fold} must contain all four classes.")
    return development


def extract_features(data_root, rows):
    """Decode each development image once and reuse its features across folds."""
    histograms = np.empty((len(rows), 3, 32), dtype=np.float32)
    gradients = np.empty((len(rows), 2), dtype=np.float32)
    for i, (path, _, _) in enumerate(rows):
        try:
            with Image.open(data_root / path) as image:
                rgb = image.convert("RGB")
                pixels = np.asarray(rgb)
                gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255
        except OSError as exc:
            raise ValueError(f"Could not read development image {path}: {exc}") from exc
        for channel in range(3):
            counts = np.bincount((pixels[:, :, channel] >> 3).ravel(), minlength=32)
            histograms[i, channel] = counts / gray.size
        magnitude = np.hypot(
            sobel(gray, axis=0, mode="reflect"),
            sobel(gray, axis=1, mode="reflect"),
        )
        gradients[i] = magnitude.mean(), magnitude.std()
    return {
        16: np.column_stack((histograms.reshape(len(rows), 3, 16, 2).sum(axis=-1)
                             .reshape(len(rows), -1), gradients)),
        32: np.column_stack((histograms.reshape(len(rows), -1), gradients)),
    }


def metrics(actual, predicted):
    labels = np.arange(len(CLASSES))
    recall = recall_score(actual, predicted, labels=labels, average=None, zero_division=0)
    return {
        "n_images": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_recall": dict(zip(CLASSES, recall.tolist())),
        "per_class_support": dict(zip(CLASSES, np.bincount(actual, minlength=len(labels)).tolist())),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def evaluate(features, labels, folds):
    configs = []
    predictions_by_config = {}
    # A single nearest-neighbor search per fold and bin count serves all k values.
    with threadpool_limits(limits=2):
        for bins in BINS:
            predictions = {k: np.empty_like(labels) for k in NEIGHBORS}
            for fold in FOLDS:
                validation = folds == fold
                train = ~validation
                scaler = StandardScaler()
                train_x = scaler.fit_transform(features[bins][train])
                validation_x = scaler.transform(features[bins][validation])
                classifier = KNeighborsClassifier(
                    n_neighbors=max(NEIGHBORS), metric="euclidean", algorithm="brute"
                )
                classifier.fit(train_x, labels[train])
                neighbor_labels = labels[train][classifier.kneighbors(
                    validation_x, return_distance=False
                )]
                for k in NEIGHBORS:
                    votes = (neighbor_labels[:, :k, None] == np.arange(len(CLASSES))).sum(axis=1)
                    predictions[k][validation] = votes.argmax(axis=1)
            for k, predicted in predictions.items():
                fold_metrics = [
                    {"fold": fold, **metrics(labels[folds == fold], predicted[folds == fold])}
                    for fold in FOLDS
                ]
                config = {
                    "bins_per_channel": bins,
                    "k": k,
                    "distance": "euclidean",
                    "fold_metrics": fold_metrics,
                    "mean_fold_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
                    "mean_fold_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
                    "pooled_metrics": metrics(labels, predicted),
                }
                configs.append(config)
                predictions_by_config[bins, k] = predicted
                print(f"bins={bins}, k={k}: mean CV macro-F1={config['mean_fold_macro_f1']:.4f}",
                      flush=True)
    # Prefer the smaller configuration when the selection metric ties exactly.
    selected = max(
        configs, key=lambda c: (c["mean_fold_macro_f1"], -c["bins_per_channel"], -c["k"])
    )
    return configs, selected, predictions_by_config[selected["bins_per_channel"], selected["k"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = development_rows(args.manifest)
    print(f"Extracting histogram and Sobel features for {len(rows)} development images.",
          flush=True)
    start = perf_counter()
    features = extract_features(args.data_root, rows)
    extracted = perf_counter()
    labels = np.asarray([label for _, label, _ in rows], dtype=np.int64)
    folds = np.asarray([fold for _, _, fold in rows], dtype=np.int64)
    configs, selected, predictions = evaluate(features, labels, folds)
    report = {
        "evaluation_set": "development",
        "evaluation_protocol": "Three fixed development folds; test images are never loaded.",
        "metric_interpretation": (
            "Development tuning estimates. The selected configuration was chosen on these "
            "same folds, so its scores are not independent final-test estimates."
        ),
        "selection_metric": "mean_fold_macro_f1",
        "class_order": CLASSES,
        "confusion_matrix_axes": {"rows": "true label", "columns": "predicted label"},
        "selected_config": {
            key: selected[key] for key in ("bins_per_channel", "k", "distance")
        },
        "mean_fold_accuracy": selected["mean_fold_accuracy"],
        "mean_fold_macro_f1": selected["mean_fold_macro_f1"],
        "pooled_metrics": selected["pooled_metrics"],
        "feature_extraction_seconds": extracted - start,
        "cross_validation_seconds": perf_counter() - extracted,
        "configurations": configs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "baseline_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(("path", "fold", "true_label", "predicted_label"))
        writer.writerows(
            (path.as_posix(), fold, CLASSES[actual], CLASSES[predicted])
            for (path, actual, fold), predicted in zip(rows, predictions)
        )
    report_path = args.output_dir / "baseline_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
