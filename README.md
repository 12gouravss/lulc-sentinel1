# Land Use / Land Cover Segmentation from Sentinel-1 Radar

A deep-learning project that maps **land use and land cover (LULC)** using **Sentinel-1 Synthetic Aperture Radar (SAR) as its only input** — no optical imagery. It includes a trained model, a prototype that produces twelve monthly land-cover maps for any location, change analysis, forecasting, and an interactive web app.

**Main result:** the full-year (multi-temporal) U-Net reaches **overall accuracy 0.850** on held-out regions, from radar alone.

---

## 1. Overview

- **Task:** semantic segmentation — assign one of 7 land-cover classes to every pixel.
- **Input:** Sentinel-1 GRD radar (VV + VH polarisations) only. Works through cloud and at night, but carries no colour, which makes some classes harder to separate.
- **Classes (7):** Water, Trees, Grassland, Cropland, Built-up, Bare/Sparse, Wetland.
- **Labels:** ESA WorldCover 2021 (10 m), used as ground truth.
- **Coverage:** 12 European regions, all 12 months of 2021 (9 regions for training, 3 held out for testing).

### Headline numbers
| Model | Test accuracy (OA) | mean IoU |
|---|---|---|
| Single-month U-Net (baseline) | 0.818 | 0.421 |
| Full-year U-Net + TTA (main) | **0.850** | 0.445 |

Other results: temporal smoothing cuts noise-driven "phantom" change by **87%**; against the independent Google Dynamic World map, built-up cover matched to within **1.3 percentage points**.

---

## 2. Public links

| Item | Link |
|---|---|
| **Model weights** | https://huggingface.co/Gouravss/lulc-sentinel1/tree/main/weights |
| **Dataset** (train/test .npz) | https://huggingface.co/datasets/Gouravss/lulc-sentinel1-data |
| **Code mirror (Hugging Face)** | https://huggingface.co/Gouravss/lulc-sentinel1 |

**Main model weight:** `unet_multitemporal.pth` (full-year U-Net). The weights folder also contains `unet_s1_lulc.pth` (single-month baseline) and `lstm_temporal.pth` (temporal LSTM).

---

## 3. Requirements

Python 3.10+ and the packages in `requirements.txt`:

```
earthengine-api>=0.1.380
geemap>=0.30.0
torch>=2.0.0
numpy>=1.24.0
rasterio>=1.3.0
opencv-python>=4.8.0
matplotlib>=3.7.0
```

Install:
```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended for training (trained on an NVIDIA RTX A6000). Inference and the prototype run on CPU or GPU.

---

## 4. Required account / API key

Imagery is pulled from **Google Earth Engine (GEE)**, which needs a free account:

1. Sign up at https://earthengine.google.com and note your Cloud project ID.
2. Authenticate once:
   ```bash
   earthengine authenticate
   ```
3. Set your project ID in the scripts (the `PROJECT` variable at the top, e.g. `PROJECT = "lulc-sentinel-513912"`).

No other API keys are needed. WorldCover labels and Dynamic World (used only for the independent check) are read through the same GEE access.

---

## 5. Setup

```bash
git clone https://github.com/12gouravss/lulc-sentinel1.git
cd lulc-sentinel1
pip install -r requirements.txt
earthengine authenticate
```

Download the trained weights from the weights link above and place the `.pth` files in the project root (or a `weights/` folder and adjust paths). For training from scratch you also need the dataset (`train.npz`, `test.npz`) from the dataset link, or you can regenerate it with `server_data_pull.py`.

---

## 6. How to run

### Data pipeline (pull Sentinel-1 + WorldCover, build dataset)
```bash
python3 server_data_pull.py          # full pull (add --test for a quick 1-region check)
```

### Train the models
```bash
python3 train.py                     # single-month baseline  -> unet_s1_lulc.pth
python3 train_multitemporal.py       # full-year main model    -> unet_multitemporal.pth
python3 lstm_temporal.py             # temporal-consistency LSTM-> lstm_temporal.pth
```

### Prototype (latitude/longitude + start month -> 12 monthly maps + change)
```bash
python3 prototype.py --lat 49.4875 --lon 8.4660 --start-month 1 --year 2021
```
Outputs: `monthly_maps.png`, `class_percentages.csv`, `month_to_month_change_pp.csv`, `total_change_pp.csv`, a class-transition matrix, a change-detection map, and an independent check vs Dynamic World.

### Interactive web app (Change Explorer / Hotspot Finder / Forecast)
```bash
python3 app.py
```
Prints a public `gradio.live` link.

### Extra analyses
```bash
python3 hotspot_finder.py            # change / hotspot detection between two years
python3 forecast.py --lat 48.15 --lon 11.58 --years 2018 2019 2020 2021 2022 2023 --predict 2
python3 transfer_learning.py         # adapt the model to a new region with few labels
python3 optimizer_comparison.py      # SGD vs Adam vs AdamW
python3 dw_case_study.py --lat 49.4875 --lon 8.4660 --city Mannheim --year 2021  # vs Dynamic World
```

---

## 7. Repository structure

```
server_data_pull.py        Pull Sentinel-1 + WorldCover, patchify -> train.npz / test.npz
train.py                   Single-month U-Net trainer (resume-capable)
train_multitemporal.py     Full-year (multi-temporal) main model
lstm_temporal.py           Temporal-consistency LSTM
optimizer_comparison.py    Optimizer comparison (SGD / Adam / AdamW)
forecast.py                Change forecasting (LSTM + Markov baseline)
transfer_learning.py       Adapt the trained model to a new region with few labels
hotspot_finder.py          Change / hotspot detection
dw_case_study.py           Prototype vs Google Dynamic World sanity check
make_map_figures.py        12-month maps + prediction-vs-truth figures
prototype.py               CLI prototype (lat/lon + month -> 12 maps + %/change)
app.py                     Interactive web application (Gradio)
healthcheck.py             Verifies environment, files, and weight loading
notebooks/                 01-04 notebooks with execution results and training curves
requirements.txt           Dependencies
```

---

## 8. Method summary

- **Preprocessing:** Sentinel-1 GRD (already calibrated to dB and terrain-corrected by GEE) → monthly median composite → speckle filter → clamp to dB ranges and scale to 0–1 → 256×256 patches → labels co-registered at 10 m → random flips/rotations.
- **Model:** U-Net (~31M parameters). The main model reads per-pixel statistics across all 12 months (10 channels) so it can use seasonal radar signatures — this is why cropland accuracy jumps (IoU 0.831 → 0.914).
- **Loss / training:** weighted cross-entropy + Dice (+ focal for the main model); Adam with learning-rate reduction on plateau; early stopping; test-time augmentation.
- **Temporal LSTM:** smooths each pixel's month-by-month sequence to remove speckle-driven flips (phantom change −87%).

---

## 9. Limitations

- Accuracy is bounded by using radar alone — some class confusions (e.g. grassland vs sparse crop vs scrub) are physical, not fixable by more training.
- The 7-class scheme is deliberately matched to what radar can resolve.
- Change percentages are indicative, not exact — they inherit the classifier's error; the temporal LSTM reduces that noise substantially.
- The forward projections are illustrative "if current trends continue" estimates, not reliable predictions.

---

