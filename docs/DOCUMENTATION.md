# Argus: Technical Documentation

This is the full technical reference for **Argus**, which detects helmet and seat-belt
violations and reads number plates. For a short overview and quick start, see the
[README](../README.md). This document covers each pipeline stage, the code behind it,
configuration, outputs, evaluation and troubleshooting.

Section numbers in the form **§N** refer to the numbered sections in
[`notebooks/helmet_violation_detection.ipynb`](../notebooks/helmet_violation_detection.ipynb).

---

## Table of contents

1. [What Argus does](#1-what-argus-does)
2. [Repository layout](#2-repository-layout)
3. [System requirements](#3-system-requirements)
4. [Environment setup](#4-environment-setup)
5. [Datasets](#5-datasets)
6. [Baseline pipeline (§1–11)](#6-baseline-pipeline-111)
7. [Advanced techniques (§12)](#7-advanced-techniques-12)
8. [Rider-level evaluation and ablation (§13)](#8-rider-level-evaluation-and-ablation-13)
9. [Seat-belt detection (§14)](#9-seat-belt-detection-14)
10. [Key functions](#10-key-functions)
11. [Configuration settings](#11-configuration-settings)
12. [Outputs](#12-outputs)
13. [Results](#13-results)
14. [Running outside Colab](#14-running-outside-colab)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitations and next steps](#16-limitations-and-next-steps)

---

## 1. What Argus does

Argus takes traffic images or video and:

1. Detects riders, helmeted heads, helmetless heads and number plates with a fine-tuned
   **YOLOv8** detector.
2. Flags a **helmet violation** when a helmetless head lies inside a rider's box.
3. Finds that rider's **number plate** and reads it with **EasyOCR**. PaddleOCR is
   available as a cross-check.
4. On video, tracks each rider across frames so that every violator is counted once. It then
   combines evidence from all of that rider's frames: per-character voting, Indian-plate
   grammar and N-of-M confirmation.
5. Flags car occupants **without a seat belt**, using a second YOLOv8 model.
6. Exports annotated images and videos, plus CSV tables of violations and their plate
   readings.

---

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `notebooks/helmet_violation_detection.ipynb` | Main notebook, §1–14: training, evaluation, OCR, video, advanced techniques, ablation and seat belt. |
| `notebooks/helmet_basic.ipynb` | Baseline only: training, detection, OCR and video, without the advanced sections. Easiest to read. |
| `docs/DOCUMENTATION.md` | This file. |
| `docs/presentation_script_and_qa.md` | Talk track and viva Q&A. |
| `docs/Argus_Presentation.pptx` | Slide deck. |
| `samples/README.md` | Links to demo clips and trained weights, which are kept out of git. |
| `requirements.txt` | Python packages for running outside Colab. |
| `.gitignore` | Excludes credentials, weights, datasets, videos, CSVs and output folders. |

---

## 3. System requirements

**Recommended runtime: Google Colab with a T4 GPU.**

- **GPU:** an NVIDIA CUDA GPU. A Colab T4 is enough, and helmet training takes about
  2 minutes. YOLOv8, EasyOCR and PaddleOCR all use CUDA when it is available.
- **No TPU.** Do not pick a TPU runtime such as `v5e-1`. Ultralytics, EasyOCR and OpenCV do
  not run on TPUs, so the work falls back to the CPU and the TPU sits idle. Apple-silicon
  Macs have no CUDA either, so local runs are CPU-only and slow.
- **Python:** 3.10 or later. The Colab default is fine.

### Dependencies

| Package | Role |
|---------|------|
| `ultralytics` | YOLOv8 training, detection and tracking (ByteTrack via `model.track`) |
| `easyocr` | Main plate OCR engine |
| `paddleocr` / `paddlepaddle` | Optional second OCR engine, for the cross-check |
| `kagglehub` / `kaggle` | Helmet dataset download |
| `roboflow` | Seat-belt dataset download (installed in §14) |
| `opencv-python` | Image and video input/output, cropping, deskewing and perspective correction |
| `opencv-contrib-python` | Optional: FSRCNN super-resolution (`cv2.dnn_superres`) in §12.1 |
| `pyyaml` | Reading and writing `data.yaml` |
| `pandas`, `numpy`, `matplotlib` | Result tables, array maths and plots |

---

## 4. Environment setup

### Google Colab (recommended)

1. Open `notebooks/helmet_violation_detection.ipynb` in
   [Colab](https://colab.research.google.com), using **File → Open notebook → GitHub** or
   **File → Upload notebook**.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Open the **Secrets** panel (🔑 in the left sidebar), add the keys below and allow the
   notebook to access them:

   | Secret | Needed for | Where to get it |
   |--------|------------|-----------------|
   | `KAGGLE_USERNAME`, `KAGGLE_KEY` | Helmet dataset (§2) | kaggle.com → Settings → API → Create New Token |
   | `ROBOFLOW_API_KEY` | Seat-belt dataset (§14) only | roboflow.com → Settings → API Keys |

   §2 copies the Kaggle secrets into environment variables for the `kaggle` CLI. If no
   secrets are found, it prints a warning, and you can upload `kaggle.json` instead.
4. Select **Runtime → Run all**, or run the sections in order.

> **`helmet_basic.ipynb` does not read Colab Secrets.** Before its download cell, either run
> this cell:
>
> ```python
> from google.colab import files
> files.upload()                      # select kaggle.json
> !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
> ```
>
> or set `os.environ["KAGGLE_USERNAME"]` and `os.environ["KAGGLE_KEY"]` yourself.

### Check the runtime

```python
import os
try:
    import google.colab; env = "Google Colab"
except ImportError:
    env = "Local / other"
print("Environment :", env, "| cwd:", os.getcwd())
import torch
print("CUDA        :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

On Colab you should see `Google Colab`, `/content`, `True` and `Tesla T4`.

---

## 5. Datasets

### Helmet and plates (Kaggle)

[Rider with helmet / without helmet / number plate](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate).
§2 downloads it, unzips it into `dataset/` and writes `dataset/data.yaml`:

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

Class IDs are **not hard-coded** in later cells. `find_id(...)` looks each one up by name or
alias (for example `"rider"` or `"motorcyclist"`), so a dataset that uses the same names in a
different order still works.

### Seat belt (Roboflow Universe)

This is a separate YOLO-format dataset with the classes `with-seatbelt` and
`Without-seatbelt`. §14.1 downloads it through the Roboflow API. To use it, set `WORKSPACE`,
`PROJECT` and `VERSION` from the dataset's "Download → YOLOv8" snippet. The notebook lists
some candidate datasets. To use your own data instead, put it at
`/content/seatbelt_dataset/data.yaml` (the `SEATBELT_DATA_YAML` setting).

---

## 6. Baseline pipeline (§1–11)

| § | Stage | Details |
|---|-------|---------|
| 1 | Install | `pip install ultralytics kagglehub easyocr pyyaml` |
| 2 | Download | Load the Kaggle secrets, download and unzip the dataset, print image and label counts, write `data.yaml`. |
| 3 | Train | `YOLO("yolov8n.pt").train(data="/content/dataset/data.yaml", epochs=50, imgsz=640, batch=16)`, then show `results.png`. |
| 4 | Validate | `model.val()` reports precision, recall, mAP50 and mAP50-95, overall and per class. |
| 5 | Rider-level violations | Resolve `RIDER`, `NO_HELMET` and `PLATE` with `find_id`. A rider is a violation when the **centre of a `without helmet` box lies inside the rider box** (`center_in`). Predicted violators are matched to ground-truth riders from the YOLO labels (`load_gt`) at **IoU ≥ 0.5**. |
| 6 | Plate crop and OCR experiments | Crop plates, upscale them 5× and try OCR on one plate (`plate_crops/`) and then on every validation plate (`plate_results/`). |
| 7 | PaddleOCR (optional) | Compare against EasyOCR. It is unreliable on Colab, so it is wrapped in `try/except`. |
| 8 | Deskew | `get_angle` estimates the plate's tilt from Canny edges and `HoughLinesP` and rotates the crop straight. Perspective correction with `getPerspectiveTransform` is also tried. |
| 9 | OCR scoring | `generate_ocr_candidates` upscales the crop 4× and builds five versions: original, grayscale, sharpened, Otsu threshold and adaptive threshold. It runs EasyOCR on each with an `A–Z0–9` allowlist. `candidate_score` ranks the readings, and the best per plate goes to `final_ocr_results.csv`. |
| 10 | End to end | `analyse_image` finds each helmetless rider, matches the rider to a plate (`plate_for_rider`), reads the plate (`read_plate`) and draws the result (`draw_violations`). Images go to `violations/` and the table to `violations.csv`. A cell then copies `best.pt` to Drive as `helmet_plate_yolov8_best.pt`. |
| 11 | Your own video | Can run on its own. It loads weights from Drive, or from this session's `runs/` if there is no Drive copy, redefines the helpers, and uploads a clip. The video loop uses `model.track(..., persist=True)`, resizes to 1280×720 and processes every second frame (`FRAME_SKIP = 2`). It keeps the best plate reading per track ID and writes `annotated_output.mp4`, `video_violations.csv` and per-rider crops in `plate_evidence/`. Readings below `MIN_CONF = 0.30` are recorded as **NO PLATE**. |

---

## 7. Advanced techniques (§12)

The baseline follows the standard **YOLO + tracking + EasyOCR** approach. §12 adds six
methods that each produce a number you can compare before and after. First run the setup and
upload cells in §11, then run §12.1–12.6 in order.

| § | Technique | Baseline | What §12 does instead |
|---|-----------|----------|-----------------------|
| 12.1 | **Super-resolution and sharpest-frame OCR** | Cubic resize on every frame, blurred or not | 4× FSRCNN upscaling (`upscale`), plus `sharpness` (variance of the Laplacian) to pick the sharpest frame. If `opencv-contrib` is missing, it falls back to cubic resize. |
| 12.2 | **Grammar-constrained decoding** | One O→0 / I→1 swap applied to the whole string, which breaks state letters | `correct_indian_plate` corrects each position against `SS DD L(1-3) NNNN`: letter↔digit maps per position and a list of real state codes. Returns `(fixed, is_valid)`. |
| 12.3 | **Confidence-weighted voting** | Keep the single best frame | `read_plate_chars` returns per-character confidences. `vote_plate` takes the most common plate length, then votes on each character, weighting each vote by its confidence. |
| 12.4 | **N-of-M violation confirmation** | One helmetless frame is enough to flag a rider | Flag a rider only if the track is seen in **≥ 5** frames (`CONFIRM_MIN_FRAMES`) and is helmetless in **≥ 50%** of them (`CONFIRM_RATIO`). |
| 12.4 | **Virtual tripwire and review queue** | Count by ID and trust every reading | Count each violator once when they cross the line at `LINE_Y = 0.6 × height`. Send readings with vote confidence below 0.45 (`REVIEW_CONF`) or an invalid format to **NEEDS REVIEW**. |
| 12.5 | **Two-engine cross-check** | One OCR engine | Compare EasyOCR (voted and grammar-corrected) with PaddleOCR on each rider crop. Agreement ≥ 0.8 means `trust = high`. |
| 12.6 | **CER evaluation** | — | Type in the true plates in `GT = {track_id: plate}` to get exact-match accuracy and mean Character Error Rate. |

§12.4 writes `annotated_advanced.mp4`, `violations_advanced.csv` and crops in
`plate_evidence_adv/`. The CSV has the columns `track_id`, `frames`, `nohelm_ratio`,
`plate_raw`, `plate`, `valid_format`, `vote_conf` and `status`.

---

## 8. Rider-level evaluation and ablation (§13)

**Claim.** Frame-level mAP is the metric used in the AI City Challenge Track 5
helmet-violation papers (Aboah et al. 2023, Duong et al. 2023, Van Luong et al. 2024). It
scores boxes in individual frames. It does not measure whether the *right rider* is flagged
or whether their *plate* is read correctly. Argus therefore reports a **rider-level
protocol**: per-rider precision, recall and F1 for the violation decision, plus plate
exact-match rate and **CER**.

**Controlled design.** §13.1 runs the detector and tracker **once** and stores the evidence
for each track in `EVID`, saving crops to `evidence_crops/`. §13.3 then scores four variants
on that *same* evidence, so only the aggregation step changes between them:

| Variant | Violation flag | Plate reading |
|---------|----------------|---------------|
| V0 baseline | Any single helmetless frame | Best single frame, raw OCR |
| V1 + voting | Any single frame | Confidence-weighted character vote |
| V2 + grammar | Any single frame | Vote and Indian-plate correction |
| V3 full | N-of-M confirmation | Vote and grammar correction |

**How to run it**

1. Run §13.1 to collect the evidence.
2. Look through the crops in `evidence_crops/`.
3. Run §13.2. It prints templates for `GT_VIOL` (true violator track IDs) and `GT_PLATE`
   (true plate for each track).
4. Paste the templates into §13.3 and fill them in. Labelling 10–15 tracks on one clip is
   enough.
5. Run §13.3. It writes `ablation_results.csv` with the columns `variant`, `TP`, `FP`, `FN`,
   `precision`, `recall`, `f1`, `plate_exact` and `mean_CER`.

**Reading the table.** From V0 to V3, false positives should fall and plate exact-match should
rise as CER falls. **Report recall honestly as well:** N-of-M confirmation can miss a
violator who appears in only a few frames. For short clips, lower `CONFIRM_MIN_FRAMES`.

---

## 9. Seat-belt detection (§14)

The helmet model has no seat-belt class, and seat belts are a four-wheeler problem with
different objects in view. Argus therefore trains a **second model** instead of extending
the first.

| § | Step |
|---|------|
| 14.1 | Read `ROBOFLOW_API_KEY` from Secrets (falling back to the environment variable) and download the dataset in YOLOv8 format. |
| 14.2 | Train `yolov8n.pt` for 50 epochs at 640 px with batch size 16 and save the weights to `/content/seatbelt_best.pt`. This step is skipped if the file already exists. A commented-out line copies the weights to Drive. |
| 14.3 | Load the model and find the no-seat-belt class with `find_in(...)`, which accepts aliases such as `no seatbelt`, `without seatbelt` and `no belt`. `seatbelt_violations(frame)` returns detections of that class at `SB_CONF = 0.35`. |
| 14.4 | Run **both** models on one video in a single pass. Helmet violations are drawn in **red** and seat-belt violations in **magenta**. Output: `annotated_helmet_seatbelt.mp4`. |

**Scope.** Argus can only flag a missing seat belt if the dataset has an explicit
"no seat belt" class. A dataset with only a "seat belt present" class cannot express absence
without extra reasoning about the driver or windshield region. The tracking, voting and
evaluation pipeline in §12–13 handles helmets only. Seat belt runs as a separate detector
that could be extended in the same way.

---

## 10. Key functions

| Function | § | What it does |
|----------|---|--------------|
| `find_id(*aliases)` | 5, 11 | Maps a class name or alias, ignoring case and underscores, to its model ID. |
| `center_in(outer, inner)` | 5 | True if the centre of `inner` lies inside `outer`. Used for both violation detection and plate-to-rider matching. |
| `iou(a, b)` | 5 | Intersection over union of two boxes. |
| `violating_riders(boxes, classes)` | 5 | Riders whose box contains the centre of a `NO_HELMET` box. |
| `load_gt(lbl_path, w, h)` | 5 | Reads YOLO label files into pixel-coordinate ground truth. |
| `get_angle(img)` | 8 | Estimates plate tilt from Canny edges and Hough lines. |
| `generate_ocr_candidates(image)` | 9 | Upscales 4×, builds 5 versions, runs EasyOCR on each and returns `{text, confidence, method}`. Accepts a file path or a BGR array. |
| `candidate_score(text, confidence)` | 9 | Scores a reading: `confidence × 100`, a bonus for length ≥ 6 or ≥ 8, and a penalty for ≤ 3 characters or make names (HONDA, YAMAHA and so on). |
| `plate_for_rider(rider, plates)` | 10 | Returns the plate whose centre is inside the rider's box, otherwise the nearest plate. |
| `read_plate(crop)` | 10 | Best `(text, confidence)` for one crop. |
| `analyse_image(img, conf=0.35)` | 10 | Runs the full single-image analysis and returns violations with their plate text. |
| `draw_violations(img, violations)` | 10 | Draws boxes and plate text on the image. |
| `process_video(video_path, out_path, conf=0.3)` | 11 | Runs tracking frame by frame, removes duplicate riders by track ID and keeps the best plate per rider. |
| `plate_quality(text, conf)` | 11 | Video plate score: confidence, length, a bonus for matching the Indian-plate regex and a penalty for make names. |
| `upscale(crop)`, `sharpness(img)` | 12.1 | FSRCNN 4× upscaling (falling back to cubic) and variance of the Laplacian. |
| `correct_indian_plate(text)` | 12.2 | Grammar-based correction for each position. Returns `(fixed, is_valid)`. |
| `read_plate_chars(crop)`, `vote_plate(samples)` | 12.3 | OCR with per-character confidences, then the confidence-weighted vote across frames. |
| `cer(pred, true)` | 12.6, 13.3 | Levenshtein distance divided by the true plate's length. |
| `decide(T, use_vote, use_grammar, use_confirm)` | 13.3 | Returns `(flagged, plate)` for one track under an ablation variant. |
| `seatbelt_violations(frame)` | 14.3 | No-seat-belt detections in one frame. |

---

## 11. Configuration settings

| Setting | § | Default | Notes |
|---------|---|---------|-------|
| Base model | 3, 14.2 | `yolov8n.pt` | Try `yolov8s` or `yolov8m` for better helmet classification. |
| `epochs` / `imgsz` / `batch` | 3, 14.2 | 50 / 640 / 16 | Lower `batch` if the GPU runs out of memory. |
| Image detection `conf` | 10 | 0.35 | Used by `analyse_image`. |
| Ground-truth match IoU | 5 | 0.5 | Threshold for matching violators to ground truth. |
| OCR allowlist | 9, 12.3 | `A–Z0–9` | Limits OCR to plate characters. |
| OCR upscale | 6, 9, 12.1 | 4× to 5× | Bigger crops read better but take longer. |
| `MODEL_PATH` candidates | 11 | Drive `helmet_plate_yolov8_best.pt`, then `runs/.../best.pt` | The first path that exists is used. |
| `RIDER_CONF` / `PLATE_CONF` | 11, 12.4 | 0.25 / 0.10 | Plates are faint in video, so they are detected at a lower threshold. |
| `MIN_CONF` | 11 | 0.30 | Readings below this are recorded as NO PLATE. |
| `FRAME_SKIP`, output size | 11, 12.4, 14.4 | 2, 1280×720 | Use 1 to process every frame and 3 for faster runs. |
| `CONFIRM_MIN_FRAMES` / `CONFIRM_RATIO` | 12.4, 13.3 | 5 / 0.5 | The N-of-M rule. Lower the frame count for short clips. |
| `REVIEW_CONF` | 12.4 | 0.45 | Vote confidence below this sends a reading to review. |
| `LINE_Y` | 12.4 | 0.6 × frame height | Position of the virtual tripwire. |
| `WORKSPACE` / `PROJECT` / `VERSION` | 14.1 | Placeholders | Copy them from the Roboflow download snippet. |
| `SB_CONF` | 14.3 | 0.35 | Seat-belt detection threshold. |

---

## 12. Outputs

Everything is written to `/content/` on the Colab VM. **`/content/` is deleted when the
runtime disconnects**, so copy anything you want to keep to Drive (`/content/drive/MyDrive/`).

| Path | § | Contents |
|------|---|----------|
| `runs/detect/train/` | 3 | Training curves and metrics |
| `runs/detect/train/weights/best.pt` | 3 | Helmet and plate weights. Copied to `MyDrive/Helmet_Violation_Detection/helmet_plate_yolov8_best.pt`. |
| `plate_crops/` | 6, 8 | Single-plate OCR and deskew experiments |
| `plate_results/` | 6 | Plate crops from all validation images |
| `final_ocr_results.csv` | 9 | Best reading for each plate |
| `violations/`, `violations.csv` | 10 | Annotated violation images and their plates |
| `annotated_output.mp4` | 11 | Annotated baseline video |
| `video_violations.csv` | 11 | `track_id`, `plate`, `ocr_confidence`, `status` (readable or NO PLATE) |
| `plate_evidence/` | 11 | Best plate crop for each violating rider |
| `FSRCNN_x4.pb` | 12.1 | Downloaded super-resolution model |
| `annotated_advanced.mp4`, `violations_advanced.csv`, `plate_evidence_adv/` | 12.4 | Advanced video run, confirmed violators and the review queue |
| `evidence_crops/` | 13.1 | Crops for each track, used for hand-labelling |
| `ablation_results.csv` | 13.3 | V0–V3 rider-level metrics |
| `seatbelt_best.pt` | 14.2 | Seat-belt weights |
| `annotated_helmet_seatbelt.mp4` | 14.4 | Combined helmet and seat-belt video |

---

## 13. Results

### Helmet and plate detector

YOLOv8n trained for 50 epochs on a Tesla T4, taking about 2 minutes. Evaluated on the
validation split of 20 images containing 73 objects.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.917 | 0.924 | 0.918 | 0.766 |
| with helmet | 0.941 | 0.846 | 0.914 | 0.679 |
| without helmet | 0.821 | 0.917 | 0.811 | 0.655 |
| rider | 0.915 | 0.935 | 0.952 | 0.877 |
| number plate | 0.991 | 1.000 | 0.995 | 0.852 |

At confidence 0.35, violation detection found **9 of 11** true violations, with 2 false
alarms and 2 misses. Precision, recall and F1 are each **0.818**.

### Seat-belt detector

mAP50 ≈ 0.995 overall. The violation class (`Without-seatbelt`) has mAP50-95 ≈ 0.925 and
recall 1.0.

### Ablation (§13)

The V0–V3 table depends on the clip, because it needs a hand-labelled video. Run §13 on your
own footage and report the change from V0 to V3 in false positives, plate exact-match and CER,
together with the change in recall.

> Both datasets are small and clean. With only 20 validation images, these numbers are noisy
> and indicate the approach works, not that it is ready for production.

---

## 14. Running outside Colab

The notebooks assume Colab. They use `!pip` and `!kaggle` shell cells, `google.colab`
(`userdata`, `drive`, `files`) and hard-coded `/content/...` paths. To run locally:

1. `pip install -r requirements.txt`. For §12.1, also install `opencv-contrib-python`, and
   for §14 install `roboflow`.
2. Export `KAGGLE_USERNAME`, `KAGGLE_KEY` and, if needed, `ROBOFLOW_API_KEY` as environment
   variables. §2 and §14.1 fall back to these when Colab Secrets are not available.
3. Replace `!kaggle ...` with a manual download or `kagglehub.dataset_download(...)`.
4. Change every `/content/...` path to a local folder, such as `./outputs/`.
5. Skip the Drive and `files.upload()` cells, and set `VIDEO_PATH` by hand.
6. Expect CPU speed unless you have an NVIDIA CUDA GPU.

---

## 15. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `kaggle: command not found`, or a 401 error on download | The Kaggle secrets are missing or notebook access is turned off. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Secrets, turn on access, and re-run §2. In `helmet_basic.ipynb`, upload `kaggle.json` (see §4 of this document). |
| §14.1 prints "Add ROBOFLOW_API_KEY…" | Add the secret, set `WORKSPACE`, `PROJECT` and `VERSION`, and re-run the cell. |
| `torch.cuda.is_available()` is `False` | The runtime is CPU or TPU. Switch to a T4 GPU and re-run. |
| Training is very slow | You are not on a GPU, or `batch` is too large. Use a T4 or lower `batch`. |
| `No trained model found` in §11 | Run training (§3) or the Drive save cell in §10 first, or edit the `CANDIDATES` paths. |
| `NameError` in §12 or §13 (for example `model` or `VIDEO_PATH`) | Run §11's setup and upload cells first. |
| "SR unavailable → cubic fallback" | `opencv-contrib-python` is not installed. This is harmless: §12 still runs with cubic upscaling. |
| `cannot import paddleocr`, or the runtime crashes after installing it | PaddleOCR is optional. Select **Runtime → Restart session** and re-run from the model-loading cell. §7 and §12.5 skip it automatically. |
| §13.3 shows P/R = 0 and a blank CER | `GT_VIOL` and `GT_PLATE` are empty. Fill them in using the §13.2 template. |
| V3 recall is much lower than V0 | The clip is short or riders appear in only a few frames. Lower `CONFIRM_MIN_FRAMES`. |
| OCR returns empty or nonsense text | The plate is small, blurred or angled. Make sure super-resolution is active, use `FRAME_SKIP = 1`, or try footage with larger plates. |
| `FileNotFoundError` on `/content/...` | You are running locally. Change the paths (see §14 of this document). |
| Outputs disappeared | The runtime disconnected and `/content/` was deleted. Save to a mounted Drive. |

---

## 16. Limitations and next steps

- **Small validation set.** With only 20 images, metrics are noisy, and recall near 1.0 will
  not carry over to real CCTV footage.
- **OCR on hard plates.** Small, blurred or angled plates remain unreliable. A dedicated
  plate-recognition model, such as a CRNN or LPRNet trained on Indian plates, is the next
  step.
- **Detector capacity.** `yolov8s` or `yolov8m` trained for longer should improve
  helmet-versus-no-helmet classification.
- **Tracking.** IDs can swap in dense traffic, which mixes evidence between riders.
  N-of-M confirmation trades some recall for fewer false alarms. Both effects are measured
  in §13.
- **Seat belt.** It is detected but not yet tracked, voted or evaluated at the rider level.
  The next step is a per-vehicle version of §13.
- **Demo footage.** One clip rarely suits both models: the helmet model needs two-wheeler
  footage, and the seat-belt model needs close views through the windshield.

---

## Author

Sanskar Shekhar ([@shekharsanskar9](https://github.com/shekharsanskar9))
