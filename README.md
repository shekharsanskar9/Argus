# Argus

**Traffic-violation detection with number-plate OCR.** Argus runs YOLOv8 on traffic
images and video to:

- flag two-wheeler riders **without a helmet** and read their **number plate**, and
- flag car occupants **without a seat belt** (a second detector).

Everything runs in Google Colab notebooks on a free T4 GPU.

| | |
|---|---|
| **Helmet / plate detector** | YOLOv8n, mAP50 0.918, mAP50-95 0.766 (val split) |
| **Violation detection** | Precision, recall and F1 all 0.818 (9 of 11 violations caught) |
| **Seat-belt detector** | YOLOv8, mAP50 ≈ 0.995, violation-class recall 1.0 |
| **OCR** | EasyOCR, plus deskewing, multi-variant preprocessing and Indian-plate grammar |

## Quick start (Colab)

1. Open [`notebooks/helmet_violation_detection.ipynb`](notebooks/helmet_violation_detection.ipynb)
   in Google Colab.
2. Choose a GPU runtime under **Runtime → Change runtime type** (a T4 is enough).
3. Add your keys in the Colab **Secrets** panel (🔑 in the left sidebar). No keys are stored in
   the notebook.
   - `KAGGLE_USERNAME` and `KAGGLE_KEY`: needed to download the helmet dataset.
   - `ROBOFLOW_API_KEY`: needed only for the seat-belt dataset (Section 14).
4. Run the cells from top to bottom. Section 11 lets you upload your own video.

**Running locally:** `pip install -r requirements.txt`, export the same keys as environment
variables, and change the `/content/...` paths to local ones. Drive mounting and
`files.upload()` only work in Colab, so set `VIDEO_PATH` by hand.

## Repository layout

```
notebooks/
  helmet_violation_detection.ipynb   # main notebook: train → OCR → video → advanced → seat belt
  helmet_basic.ipynb                 # shorter version: train → detect → OCR → video
docs/
  DOCUMENTATION.md                   # full technical reference
  presentation_script_and_qa.md      # talk track and viva Q&A
  Argus_Presentation.pptx            # slide deck
samples/
  README.md                          # links to demo clips and trained weights (not in git)
requirements.txt
```

Datasets, weights, videos, CSVs and output folders are created at run time and are gitignored.

### Which notebook?

- **`helmet_violation_detection.ipynb`** is the main notebook. It covers the full pipeline,
  rider-level evaluation, the advanced temporal and grammar techniques with an ablation
  study, and the seat-belt model.
- **`helmet_basic.ipynb`** is the baseline pipeline without the extras, and the easier one
  to read from top to bottom. It does not read Colab Secrets, so upload `kaggle.json` or set
  `KAGGLE_USERNAME` and `KAGGLE_KEY` before running the download cell.

## Datasets

**Helmet and plates:** [Rider with helmet / without helmet / number plate](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate) (Kaggle).

| ID | Class          |
|----|----------------|
| 0  | with helmet    |
| 1  | without helmet |
| 2  | rider          |
| 3  | number plate   |

**Seat belt:** a YOLO-format dataset from Roboflow Universe with two classes,
`with-seatbelt` and `Without-seatbelt`. You can also point `SEATBELT_DATA_YAML` at your own
dataset.

## How it works

The section numbers below match the main notebook.

| § | Stage | What happens |
|---|-------|--------------|
| 1–2 | Setup | Install `ultralytics`, `kagglehub`, `easyocr` and `pyyaml`, download the dataset, write `dataset/data.yaml`. |
| 3–4 | Train & validate | Fine-tune `yolov8n.pt` for 50 epochs at 640 px with batch size 16, then report per-class precision, recall and mAP. |
| 5 | Violation logic | A rider is a violation when the centre of a `without helmet` box falls inside the rider's box. Predictions are matched to ground truth at IoU ≥ 0.5. |
| 6, 8 | Plate crops & deskew | Crop each `number plate` detection, estimate its angle with Canny edges and Hough lines, and rotate it straight. Perspective correction is also tried. |
| 7 | PaddleOCR (optional) | Cross-check against EasyOCR. It is unreliable on Colab, so it is wrapped in `try/except`. |
| 9 | OCR scoring | Upscale 4× and build grayscale, sharpened, Otsu and adaptive-threshold variants. Run EasyOCR on each with an `A-Z0-9` allowlist. Candidates are ranked by confidence, length and an Indian-plate regex, and make names such as HONDA and YAMAHA are penalised. |
| 10 | End to end | `analyse_image` links each helmetless rider to a plate (the plate inside the rider's box, otherwise the nearest one), reads it, and saves annotated images. |
| 11 | Video | `process_video` uses YOLO tracking so each violator is counted once, and keeps the best plate reading for each track. |
| 12–13 | Advanced techniques | See below. |
| 14 | Seat belt | A second YOLOv8 detects `Without-seatbelt`. Both models can run on one video in a single pass: helmet violations are drawn in red and seat-belt violations in magenta. |

### Advanced techniques (the research contribution)

Published helmet-detection work usually reports per-frame mAP, which does not measure how
well a system enforces the rules end to end. On top of the baseline, the main notebook adds:

- **Rider-level evaluation:** per-rider precision and recall, plus plate Character Error
  Rate (CER), not just mAP.
- **Temporal evidence aggregation:**
  - confidence-weighted character voting across the frames of each tracked rider;
  - grammar-constrained decoding against the Indian plate format (`SS DD L(1-3) NNNN`),
    checked against a list of real state codes;
  - N-of-M temporal confirmation, so a violation that appears in only one frame does not
    raise an alert.
- **Ablation study** (baseline → voting → grammar → N-of-M) to measure what each step adds,
  plus a human-review queue for low-confidence readings.

See Sections 12–13 of the notebook and [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

## Results

The helmet model was trained as YOLOv8n for 50 epochs on a Tesla T4, which took about
2 minutes. Results are on the validation split of 20 images containing 73 objects.

| Class          | Precision | Recall | mAP50 | mAP50-95 |
|----------------|-----------|--------|-------|----------|
| all            | 0.917     | 0.924  | 0.918 | 0.766    |
| with helmet    | 0.941     | 0.846  | 0.914 | 0.679    |
| without helmet | 0.821     | 0.917  | 0.811 | 0.655    |
| rider          | 0.915     | 0.935  | 0.952 | 0.877    |
| number plate   | 0.991     | 1.000  | 0.995 | 0.852    |

At confidence 0.35, violation detection found 9 of 11 true violations, with 2 false alarms
and 2 misses. Precision, recall and F1 are each 0.818.

The seat-belt model reaches mAP50 ≈ 0.995 overall. Its violation class (`Without-seatbelt`)
has mAP50-95 ≈ 0.925 and recall 1.0.

> Both datasets are small and clean, so treat these numbers as indicative, not
> production-grade.

## Outputs

All outputs are written under `/content/` in Colab.

| Path | Contents |
|------|----------|
| `runs/detect/train/` | Training curves and metrics |
| `runs/detect/train/weights/best.pt` | Best helmet and plate weights (optionally copied to Drive) |
| `plate_crops/` | Single-plate OCR experiments |
| `plate_results/` | Plate crops from all validation images |
| `final_ocr_results.csv` | Best reading for each plate |
| `violations/`, `violations.csv` | Annotated violation images and their plates |
| `annotated_output.mp4`, `video_violations.csv`, `plate_evidence/` | Video run: annotated clip, violation log, plate crops |
| `annotated_advanced.mp4`, `violations_advanced.csv` | Video run with the advanced techniques |
| `ablation_results.csv` | Ablation table (Section 13) |
| `seatbelt_best.pt` | Seat-belt weights |
| `annotated_helmet_seatbelt.mp4` | Combined helmet and seat-belt video |

## Limitations and next steps

- With only 20 validation images, the metrics are noisy. Recall near 1.0 will not carry
  over to real CCTV footage.
- OCR is still unreliable on small, blurred or angled plates. A dedicated plate-recognition
  model or higher-resolution crops would help.
- A larger model (`yolov8s` or `yolov8m`) trained for longer should improve helmet
  classification.
- Tracking can swap IDs in dense traffic. N-of-M confirmation can miss a violator who
  appears in very few frames, which is a measured trade-off against recall.
- One clip rarely shows both cases well: the helmet model needs two-wheeler footage, and the
  seat-belt model needs close views through the windshield.

## Security

Keys are read from Colab Secrets or environment variables. Never commit `kaggle.json`, API
keys, trained weights or generated outputs. `.gitignore` already excludes them.

## Author

Sanskar Shekhar ([@shekharsanskar9](https://github.com/shekharsanskar9))
