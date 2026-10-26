# Argus — Technical Documentation

Full technical reference for the **Argus** helmet-violation detection and
number-plate OCR project. For a quick overview, see [README.md](README.md);
this document goes deeper into architecture, each pipeline stage, the code that
implements it, configuration, outputs, and troubleshooting.

---

## Table of contents

1. [What Argus does](#1-what-argus-does)
2. [Repository layout](#2-repository-layout)
3. [System requirements](#3-system-requirements)
4. [Environment setup](#4-environment-setup)
5. [Dataset](#5-dataset)
6. [Pipeline reference](#6-pipeline-reference)
7. [Key functions](#7-key-functions)
8. [Configuration knobs](#8-configuration-knobs)
9. [Outputs](#9-outputs)
10. [Results](#10-results)
11. [Running outside Colab](#11-running-outside-colab)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations and next steps](#13-limitations-and-next-steps)

---

## 1. What Argus does

Argus takes traffic images (or video) and:

1. Detects riders, helmets, "no-helmet" heads, and number plates with a
   fine-tuned **YOLOv8** object detector.
2. Flags a **helmet violation** when a rider is detected without a helmet.
3. Locates that rider's **number plate** and reads it with **OCR**
   (EasyOCR, with PaddleOCR available for comparison).
4. Exports annotated images and a table of violations with their plate
   readings.

The whole pipeline is packaged as a single Google Colab notebook,
`helmet_detection_colab.ipynb`, meant to be run top to bottom on a GPU runtime.

---

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `helmet_detection_colab.ipynb` | End-to-end pipeline: train → evaluate → detect violations → OCR → export. The main notebook. |
| `dl_training_colab.ipynb` | General deep-learning training workflow (supporting notebook). |
| `README.md` | Short project overview and quick-start. |
| `DOCUMENTATION.md` | This file. |
| `requirements.txt` | Python packages for running outside Colab. |
| `.gitignore` | Excludes `kaggle.json`, trained weights, and generated output folders. |

---

## 3. System requirements

**Recommended runtime: Google Colab with a T4 GPU.** The project is built on
PyTorch/CUDA and CPU tooling.

- **GPU:** an NVIDIA CUDA GPU (Colab T4 is enough; training ~2 min). YOLOv8
  training, EasyOCR, and PaddleOCR all use CUDA when available.
- **Do not use a TPU runtime** (e.g. `v5e-1`). Ultralytics YOLOv8, EasyOCR,
  and OpenCV do not run on TPUs — the TPU would sit idle while the work falls
  back to CPU. Apple-silicon Macs have no CUDA, so local runs are CPU-only and
  slow.
- **Python:** 3.10+ (Colab default is fine).

### Core dependencies

| Package | Role |
|---------|------|
| `ultralytics` | YOLOv8 detection and training |
| `easyocr` | Primary plate OCR engine |
| `paddleocr` / `paddlepaddle` | Optional second OCR engine, for comparison |
| `kagglehub` / `kaggle` | Dataset download |
| `opencv-python` (`cv2`) | Image I/O, crop, deskew, perspective correction |
| `pyyaml` | Writes/reads `data.yaml` |
| `pandas`, `numpy` | Results tables and array math |

---

## 4. Environment setup

### In Google Colab (recommended)

1. Open `helmet_detection_colab.ipynb` in
   [Colab](https://colab.research.google.com) (**File → Upload notebook**, or
   **File → Open notebook → GitHub**).
2. **Runtime → Change runtime type → T4 GPU.**
3. Provide your Kaggle API token so the dataset can download. Create it at
   **kaggle.com → Settings → API → Create New Token**, then run this cell
   *before* the dataset-download cell:

   ```python
   from google.colab import files
   files.upload()                      # select kaggle.json
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Runtime → Run all.**

### Verify where it is running

Drop this at the top to confirm environment and GPU:

```python
import sys, os, platform
try:
    import google.colab; env = "Google Colab"
except ImportError:
    env = "Local / other"
print("Environment :", env)
print("Working dir :", os.getcwd())
try:
    import torch
    print("CUDA avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU         :", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed yet")
```

Expect `Google Colab`, working dir `/content`, `CUDA avail: True`, and a
`Tesla T4`.

---

## 5. Dataset

[Rider with helmet / without helmet / number plate](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate)
from Kaggle, four classes:

| ID | Class |
|----|-------|
| 0 | with helmet |
| 1 | without helmet |
| 2 | rider |
| 3 | number plate |

The notebook downloads and unzips it to `dataset/`, then writes
`dataset/data.yaml`:

```yaml
path: /content/dataset
train: train/images
val: val/images
names:
  0: with helmet
  1: without helmet
  2: rider
  3: number plate
```

Class IDs are **not hard-coded downstream** — the notebook resolves them by
name via `find_id(...)` (see [Key functions](#7-key-functions)), so a dataset
with the same class names in a different order still works.

---

## 6. Pipeline reference

Stages, in notebook order:

1. **Setup** — install `ultralytics`, `kagglehub`, `easyocr`, `pyyaml`.
2. **Download** — pull and unzip the Kaggle dataset; write `data.yaml`.
3. **Train** — fine-tune `yolov8n.pt`:
   ```python
   model = YOLO("yolov8n.pt")
   results = model.train(data="/content/dataset/data.yaml",
                         epochs=50, imgsz=640, batch=16)
   ```
4. **Evaluate** — `model.val()` reports precision, recall, mAP50, mAP50-95
   overall and per class.
5. **Resolve classes** — `RIDER`, `NO_HELMET`, `PLATE` via `find_id(...)`.
6. **Detect violations** — a rider is a violation when the **centre of a
   `without helmet` box falls inside the rider box** (`center_in`). Predicted
   violations are matched to ground truth at **IoU ≥ 0.5** (`iou`).
7. **Crop plates** — every `number plate` detection in the val images is
   upscaled 5× and saved to `/content/plate_results/`.
8. **Deskew / correct** — single-plate experiments (`/content/plate_crops/`)
   estimate plate angle with Canny edges + Hough lines and rotate the crop
   straight; perspective correction via `getPerspectiveTransform` is also tried.
9. **OCR** — `generate_ocr_candidates` upscales 4× and produces grayscale,
   sharpened, Otsu, and adaptive-threshold variants, running EasyOCR on each
   with an `A–Z0–9` allowlist. Candidates are ranked by `candidate_score`
   (confidence + length, penalising manufacturer words like HONDA/YAMAHA).
10. **Export** — best reading per plate → `/content/final_ocr_results.csv`.
11. **End to end** — `analyse_image` finds each no-helmet rider, matches it to
    its plate (`plate_for_rider`), reads the plate (`read_plate`), draws
    annotations (`draw_violations`). Images → `/content/violations/`, table →
    `/content/violations.csv`.
12. **Video** — `process_video` runs the same per frame with YOLO tracking so
    each violating rider is counted once and the best plate reading is kept.
13. **PaddleOCR (optional)** — a comparison against EasyOCR.

---

## 7. Key functions

| Function | Cell area | What it does |
|----------|-----------|--------------|
| `find_id(*aliases)` | class resolution | Maps a class name (any alias, case/underscore-insensitive) to its model ID. Used to set `RIDER`, `NO_HELMET`, `PLATE`. |
| `center_in(outer, inner)` | violation logic | True if the centre of `inner` lies inside `outer`. Core of violation and plate-to-rider matching. |
| `iou(a, b)` | evaluation | Intersection-over-union of two boxes; ground-truth matching uses IoU ≥ 0.5. |
| `violating_riders(boxes, classes)` | violation logic | Riders that contain a `NO_HELMET` head centre. |
| `generate_ocr_candidates(image)` | OCR | Upscales 4×, builds 5 image variants (original/gray/sharp/otsu/adaptive), runs EasyOCR on each, returns `{text, confidence, method}` candidates. Accepts a path or a BGR array. |
| `candidate_score(text, confidence)` | OCR ranking | `confidence*100`, bonus for length ≥ 6/8, penalty for ≤ 3 chars and for manufacturer words. |
| `plate_for_rider(rider, plates)` | end to end | Plate whose centre is inside the rider box, else the nearest plate. |
| `read_plate(crop)` | end to end | Best `(text, confidence)` for a crop via the candidate pipeline. |
| `analyse_image(img)` | end to end | Full single-image analysis → list of violations with plate text. |
| `draw_violations(img, violations)` | end to end | Annotates boxes and plate text onto an image. |
| `process_video(video_path, out_path, conf=0.35)` | video | Per-frame tracking (`model.track(..., persist=True)`); de-duplicates riders by track ID and keeps the best plate reading per rider. |

---

## 8. Configuration knobs

| Setting | Where | Default | Notes |
|---------|-------|---------|-------|
| Base model | `YOLO("yolov8n.pt")` | `yolov8n` (nano) | Try `yolov8s`/`yolov8m` for better helmet classification. |
| Epochs | `model.train(epochs=...)` | 50 | |
| Image size | `imgsz` | 640 | |
| Batch size | `batch` | 16 | Lower if you hit GPU OOM. |
| Detection confidence | `conf` | 0.35 | Used in violation detection and video. |
| GT match threshold | `iou(...) ≥ 0.5` | 0.5 | Violation-to-ground-truth matching. |
| OCR allowlist | `allowlist="A–Z0–9"` | — | Restricts OCR to plate characters. |
| OCR upscale | `cv2.resize(fx/fy=...)` | 4×–5× | Larger crops read better but cost time. |
| Video paths | `VIDEO_PATH`, `OUT_PATH` | `/content/traffic.mp4`, `/content/traffic_annotated.mp4` | Upload your own video to `VIDEO_PATH`. |

---

## 9. Outputs

All paths are on the Colab VM under `/content/` (and `runs/` in the working
dir). **`/content/` is wiped when the runtime disconnects** — mount Google
Drive to keep results.

| Path | Contents |
|------|----------|
| `runs/detect/train/` | Training curves and metrics |
| `runs/detect/train/weights/best.pt` | Best trained weights |
| `/content/plate_crops/` | Single-plate OCR experiments |
| `/content/plate_results/` | Plate crops from all val images |
| `/content/final_ocr_results.csv` | Final plate readings |
| `/content/violations/` | Annotated violation images |
| `/content/violations.csv` | Violations with their plates |
| `/content/traffic_annotated.mp4` | Annotated video (video cell) |

To persist outputs:

```python
from google.colab import drive
drive.mount('/content/drive')
# then save to /content/drive/MyDrive/Argus/ instead of /content/
```

---

## 10. Results

Validation split: 20 images, 73 objects, YOLOv8n after 50 epochs on a Tesla T4
(~2 min training).

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.917 | 0.924 | 0.918 | 0.766 |
| with helmet | 0.941 | 0.846 | 0.914 | 0.679 |
| without helmet | 0.821 | 0.917 | 0.811 | 0.655 |
| rider | 0.915 | 0.935 | 0.952 | 0.877 |
| number plate | 0.991 | 1.000 | 0.995 | 0.852 |

Violation detection at confidence 0.35 found **9 of 11** true violations, with
2 false alarms and 2 misses — precision, recall, and F1 all **0.818**.

> Note: with only 20 validation images these numbers are noisy.

---

## 11. Running outside Colab

The notebook assumes a Colab environment: shell cells use `!pip` / `!kaggle`,
and every save path is hard-coded to `/content/...`. To run locally:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Replace the `!kaggle ...` download with a manual dataset download, or use
   `kagglehub.dataset_download(...)`.
3. Change every `/content/...` path to a local folder (e.g. `./outputs/`).
4. Expect CPU-only speed unless you have an NVIDIA CUDA GPU (Apple-silicon Macs
   have no CUDA).

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `kaggle: command not found` or 401 on download | `kaggle.json` not uploaded/placed. Re-run the token-setup cell (see §4). |
| `torch.cuda.is_available()` is `False` in Colab | Runtime is CPU or TPU. Switch to **T4 GPU** and re-run. |
| Training extremely slow | Not on a GPU (TPU does not work here), or batch too large. Use T4; lower `batch`. |
| `cannot import paddleocr` | PaddleOCR is optional. **Runtime → Restart session**, then re-run from the cell that loads `best.pt`. |
| OCR returns empty/garbage text | Small, blurred, or angled plate. Increase crop upscale; rely on the deskew/perspective-correction steps. |
| `FileNotFoundError` on `/content/...` | Running locally. Change paths to local folders (see §11). |
| Outputs disappeared | Runtime disconnected; `/content/` is ephemeral. Save to mounted Drive. |

---

## 13. Limitations and next steps

- The validation set has only 20 images, so metrics are noisy.
- OCR on small, blurred, or angled plates is still unreliable. Next steps: a
  dedicated plate-recognition model or a larger crop resolution.
- Train `yolov8s.pt` or `yolov8m.pt` for longer to improve helmet
  classification.
- The video cell is untuned. Test on real footage and verify the tracker keeps
  rider IDs stable when riders overlap.

---

## Author

Sanskar Shekhar ([@shekharsanskar9](https://github.com/shekharsanskar9))
