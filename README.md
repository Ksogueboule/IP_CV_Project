# Satellite Region Classifier for Natural Land Cover

Team 6's image processing and computer vision project compares a color-histogram and Sobel-feature kNN baseline with transfer learning on ResNet-18. The four EuroSAT RGB classes are forest, herbaceous vegetation, river, and sea/lake. See `Proposal` for the approved scope and evaluation goals.

## PF3 development results

The manifest contains 11,500 images, with 9,775 development images and 1,725 held-out images. The decoded-pixel check found no exact duplicates. The baseline selected 16 histogram bins per channel and k = 3 using the development folds.

| Model and evaluation | Images | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Selected kNN, pooled three-fold development predictions | 9,775 | 93.65% | 0.9349 |
| Selected kNN, development fold 0 | 3,258 | 93.49% | 0.9330 |
| ResNet-18 after one head-only epoch, development fold 0 | 3,258 | 95.98% | 0.9602 |
| ResNet-18 after one additional layer4 + head epoch, development fold 0 | 3,258 | 98.93% | 0.9891 |

Across the three kNN development folds, 205 of 2,125 river images were classified as herbaceous vegetation. On a 200-image sample from development fold 0, kNN accuracy was 93.5% on clean images, 47.5% at 0.7× brightness, 59.0% at 1.3× brightness, and 90.0% and 77.0% at Gaussian blur σ = 1 and 2 pixels.

## Reproduce the development runs

1. Create a Python 3.12 environment and install `requirements.txt`.
2. Extract the bundled `data/EuroSAT_RGB.zip` from the repository root with `python -m zipfile -e data/EuroSAT_RGB.zip data`. Skip extraction if `data/EuroSAT_RGB/Forest` already exists from a complete extraction.
3. Run these commands at the repository root.

```text
python manifest.py --data-root data/EuroSAT_RGB --output-dir artifacts/manifest
python baseline.py --manifest artifacts/manifest/manifest.csv --data-root data/EuroSAT_RGB --output-dir artifacts/baseline
```

The manifest checks exact decoded-image duplicates. Verified geographic separation and near-duplicate screening remain open. The baseline fits its scaler on each training fold and tunes bins and k by development macro-F1. The paired robustness pilot below uses `bins_per_channel` and `k` from the `selected_config` in `baseline_metrics.json`.

```text
python dev_stress.py --manifest artifacts/manifest/manifest.csv --data-root data/EuroSAT_RGB --output-dir artifacts/dev-stress --fold 0 --bins 16 --k 3
```

The pilot tests both brightness levels and both blur levels on the same development images and records changes from their clean versions.

For the CNN pilot, install a matching PyTorch and Torchvision build from the [PyTorch installer](https://pytorch.org/get-started/locally/). A CUDA build can use the GPU. This command trains one head epoch and one last-block epoch on the same development fold used for the pilot comparison.

```text
python cnn_pilot.py --manifest artifacts/manifest/manifest.csv --data-root data/EuroSAT_RGB --output-dir artifacts/cnn-pilot --fold 0 --head-epochs 1 --finetune-epochs 1
```

All current metrics are developmental. Preserve the held-out partition until the pipeline and evaluation protocol are frozen. For the final report, run all planned CNN folds and preprocessing comparisons, complete the boundary-case review, and record matched inference timing before drawing conclusions against the held-out targets.

## Dataset attribution

The bundled ZIP is the original EuroSAT RGB release by Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. The [published archive](https://zenodo.org/records/7711810) contains 27,000 JPEGs across ten classes and has MD5 `f46e308c4d50d4bf32fedad2d3d62f3b`. The project uses the four classes listed above. EuroSAT is distributed under the [MIT license](data/EuroSAT-LICENSE.txt).

Contains modified Copernicus Sentinel data (EuroSAT release, 2018). See the [Copernicus Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
