"""Train and measure a ResNet-18 pilot on one EuroSAT development fold."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


CLASSES = ("Forest", "HerbaceousVegetation", "River", "SeaLake")


def development_rows(manifest, fold):
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "group_id", "split", "fold"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must have path, label, group_id, split, and fold columns")
        rows = [row for row in reader if row["split"] == "development"]
    if any(row["fold"] not in ("0", "1", "2") for row in rows):
        raise ValueError("Each development image must belong to fold 0, 1, or 2")
    train = [row for row in rows if int(row["fold"]) != fold]
    validation = [row for row in rows if int(row["fold"]) == fold]
    for subset in (train, validation):
        if {row["label"] for row in subset} != set(CLASSES):
            raise ValueError("Training and validation must each contain all four approved classes")
        if any(not row["group_id"] for row in subset):
            raise ValueError("Every development image must have a group ID")
    if {row["group_id"] for row in train} & {row["group_id"] for row in validation}:
        raise ValueError("A duplicate group crosses training and validation")
    if {row["path"] for row in train} & {row["path"] for row in validation}:
        raise ValueError("An image path crosses training and validation")
    return train, validation


def load_subset(rows, data_root, device):
    # Caching uint8 64x64 patches keeps repeated image reads out of the pilot.
    pixels = np.empty((len(rows), 64, 64, 3), dtype=np.uint8)
    labels = []
    for index, row in enumerate(rows):
        relative = Path(row["path"])
        if relative.anchor or ".." in relative.parts:
            raise ValueError(f"Expected a relative image path: {relative}")
        with Image.open(data_root / relative) as image:
            if image.size != (64, 64) or image.mode != "RGB":
                raise ValueError(f"Expected a 64x64 RGB image: {relative}")
            pixels[index] = np.asarray(image)
        labels.append(CLASSES.index(row["label"]))
    return torch.from_numpy(pixels).to(device), torch.tensor(labels, device=device)


def inputs(pixels, mean, std, training):
    batch = pixels.permute(0, 3, 1, 2).float().div_(255)
    if training:
        for axis in (-1, -2):
            flip = torch.rand((batch.shape[0], 1, 1, 1), device=batch.device) < 0.5
            batch = torch.where(flip, batch.flip(axis), batch)
    batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
    return batch.sub_(mean).div_(std).contiguous(memory_format=torch.channels_last)


def compute_metrics(labels, predictions):
    size = len(CLASSES)
    matrix = torch.bincount(size * labels + predictions, minlength=size * size)
    matrix = matrix.reshape(size, size).cpu().numpy()
    diagonal = matrix.diagonal()
    support = matrix.sum(axis=1)
    totals = support + matrix.sum(axis=0)
    recall = np.divide(diagonal, support, out=np.zeros(size), where=support != 0)
    f1 = np.divide(2 * diagonal, totals, out=np.zeros(size), where=totals != 0)
    return {
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "per_class_recall": dict(zip(CLASSES, recall.tolist())),
        "class_support": dict(zip(CLASSES, support.tolist())),
        "confusion_matrix": matrix.tolist(),
    }


def save_predictions(path, rows, predictions):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "label", "prediction"))
        writer.writerows((row["path"], row["label"], CLASSES[prediction])
                         for row, prediction in zip(rows, predictions.cpu().tolist(), strict=True))


def train_phase(model, phase, epochs, train, validation, validation_rows, mean, std, args, report):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    groups = [{"params": model.fc.parameters(), "lr": 0.001 if phase == "head" else 0.0001}]
    if phase == "layer4":
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)
        groups.append({"params": model.layer4.parameters(), "lr": 0.0001})
    optimizer = torch.optim.Adam(groups, weight_decay=0.0001)
    use_amp = train[0].device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = None
    checkpoint = args.output_dir / f"{phase}_best.pt"

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.eval()  # Preserve running statistics in all frozen blocks.
        model.fc.train()
        if phase == "layer4":
            model.layer4.train()
        order = torch.randperm(len(train[1]), device=train[1].device)
        loss_total = torch.zeros((), device=train[1].device)
        for indices in order.split(args.batch_size):
            batch = inputs(train[0][indices], mean, std, training=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=train[0].device.type, enabled=use_amp):
                loss = F.cross_entropy(model(batch), train[1][indices])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.detach() * len(indices)

        model.eval()
        with torch.inference_mode():
            predictions = []
            for pixels in validation[0].split(args.batch_size):
                batch = inputs(pixels, mean, std, training=False)
                with torch.autocast(device_type=batch.device.type, enabled=use_amp):
                    predictions.append(model(batch).argmax(dim=1))
            predictions = torch.cat(predictions)
        result = {
            "epoch": epoch,
            "train_cross_entropy": float(loss_total.item() / len(train[1])),
            **compute_metrics(validation[1], predictions),
            "seconds": round(time.perf_counter() - started, 2),
        }
        report["phases"][phase]["epochs"].append(result)
        if best is None or result["macro_f1"] > best["macro_f1"]:
            best = result
            report["phases"][phase]["best_validation"] = best
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "phase": phase, "fold": args.fold, "epoch": epoch}, checkpoint)
            save_predictions(args.output_dir / f"{phase}_predictions.csv", validation_rows, predictions)
        (args.output_dir / "cnn_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{phase} epoch {epoch}: accuracy={result['accuracy']:.4f}, "
              f"macro-F1={result['macro_f1']:.4f}, {result['seconds']:.1f}s", flush=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=range(3), default=0)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.head_epochs, args.finetune_epochs, args.batch_size) < 1:
        parser.error("Epochs and batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    train_rows, validation_rows = development_rows(args.manifest, args.fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading development fold {args.fold}: {len(train_rows)} training and "
          f"{len(validation_rows)} validation images on {device}", flush=True)
    train = load_subset(train_rows, args.data_root, device)
    validation = load_subset(validation_rows, args.data_root, device)
    weights = ResNet18_Weights.IMAGENET1K_V1
    normalization = weights.transforms()
    mean = torch.tensor(normalization.mean, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(normalization.std, device=device).reshape(1, 3, 1, 1)
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device=device, memory_format=torch.channels_last)
    report = {
        "scope": "Single development-fold pilot. Holdout images were not loaded or evaluated.",
        "fold": args.fold,
        "seed": args.seed,
        "class_order": CLASSES,
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "weights": weights.name,
        "input_resolution": [224, 224],
        "preprocessing": "Direct bilinear resize of the whole patch and ImageNet normalization",
        "training_augmentation": "Independent horizontal and vertical flips with probability 0.5 each",
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "phases": {phase: {"epochs": []} for phase in ("head", "layer4")},
    }
    shared = (train, validation, validation_rows, mean, std, args, report)
    head = train_phase(model, "head", args.head_epochs, *shared)
    model.load_state_dict(torch.load(head, map_location=device, weights_only=True)["model"])
    train_phase(model, "layer4", args.finetune_epochs, *shared)


if __name__ == "__main__":
    main()
