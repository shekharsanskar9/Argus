# Argus: Helmet & Seat-Belt Violation Detection + Number Plate OCR

Google Colab notebooks that process traffic footage and:

- flag two-wheeler riders **without a helmet** and read their **number plate** (OCR), and
- flag car occupants **without a seat belt** (a second detector).

## Project layout

```
notebooks/
  helmet_violation_detection.ipynb   # main pipeline (training -> OCR -> video -> seat belt)
  helmet_basic.ipynb                 # simpler end-to-end version (train -> detect -> OCR)
docs/
  DOCUMENTATION.md                   # full technical write-up
  presentation_script_and_qa.md      # talk track + viva Q&A
  Argus_Presentation.pptx            # slide deck
samples/
  README.md                          # links to demo clips / trained weights (kept out of git)
```

Runtime artefacts (`dataset/`, `runs/`, `plate_results/`, `violations/`, `*.pt`, videos) are
generated on the fly and are gitignored.

## Notebooks

- **`notebooks/helmet_violation_detection.ipynb`** — the canonical notebook. Trains a YOLOv8
  helmet detector, flags violations, reads number plates, and adds the advanced techniques
  (grammar-constrained decoding for Indian plates, confidence-weighted character voting,
  N-of-M temporal confirmation), a rider-level evaluation protocol, and a second YOLOv8 for
  seat-belt detection.
- **`notebooks/helmet_basic.ipynb`** — the simpler end-to-end version (train, detect, OCR)
  without the advanced extras. Good for reading the pipeline top to bottom.

## Dataset

[Rider with helmet / without helmet / number plate](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate)
from Kaggle. It has four classes:

| ID | Class          |
|----|----------------|
| 0  | with helmet    |
| 1  | without helmet |
| 2  | rider          |
| 3  | number plate   |

The seat-belt detector uses a separate YOLO-format dataset from Roboflow Universe
(two classes: `with-seatbelt`, `Without-seatbelt`).

## Pipeline

1. **Setup.** Install `ultralytics`, `kagglehub`, `easyocr` and `pyyaml`.
   Download and unzip the dataset, then write `dataset/data.yaml`.
2. **Train.** Fine-tune `yolov8n.pt` for 50 epochs at 640 px with batch size 16.
3. **Evaluate.** Report precision, recall and mAP per class.
4. **Detect violations.** A rider counts as a violation when the centre of a
   `without helmet` box falls inside the rider's box. Predicted violations are
   matched to ground truth at IoU ≥ 0.5.
5. **Crop plates.** Save every `number plate` detection from the validation
   images to `plate_results/`.
6. **Deskew plates.** Estimate the plate angle with Canny edges and Hough lines,
   then rotate the crop straight. Perspective correction is also tried.
7. **OCR.** Upscale each crop 4x and make grayscale, sharpened, Otsu and
   adaptive-threshold versions. Run EasyOCR on each with an `A-Z0-9` allowlist.
   Candidates are ranked by confidence and length, with manufacturer names
   such as HONDA and YAMAHA penalised. A regex score for the Indian plate format
   is also applied.
8. **Export.** Write the best reading per plate to `final_ocr_results.csv`.
9. **End to end.** `analyse_image` finds each rider without a helmet, matches it
   to its number plate (the plate inside the rider's box, else the nearest one)
   and reads the plate. Annotated images go to `violations/` and the readings
   to `violations.csv`.
10. **Video.** `process_video` does the same per frame. It uses YOLO tracking so
    each violating rider is counted once, and keeps the best plate reading seen
    for that rider.

PaddleOCR is included as an optional comparison against EasyOCR (flaky on Colab,
so it is wrapped in try/except).

## Advanced techniques (the research contribution)

Published helmet work is scored on per-frame mAP, which does not measure
end-to-end enforcement. The main notebook adds, on top of the baseline:

- a **rider-level evaluation protocol** — per-rider precision/recall plus plate
  Character Error Rate (CER), not just mAP;
- **temporal evidence aggregation** — confidence-weighted character voting across
  tracked frames, grammar-constrained decoding against the Indian plate format
  (`SS DD L(1-3) NNNN`) with a real state-code list, and N-of-M temporal
  confirmation of a violation to kill single-frame false alarms;
- an **ablation** (baseline → voting → grammar → N-of-M) measuring the effect,
  plus a human-review queue for low-confidence reads.

See Sections 12–13 of the notebook and `docs/DOCUMENTATION.md`.

## Seat-belt detection

A second YOLOv8 model (Section 14) handles the four-wheeler case: detect
`Without-seatbelt` and flag it. Both models can run on the same video in one
pass (helmet violations in red, seat-belt violations in magenta).

## Results

These are from the validation split: 20 images, 73 objects, YOLOv8n after
50 epochs on a Tesla T4. Training took about 2 minutes.

| Class          | Precision | Recall | mAP50 | mAP50-95 |
|----------------|-----------|--------|-------|----------|
| all            | 0.917     | 0.924  | 0.918 | 0.766    |
| with helmet    | 0.941     | 0.846  | 0.914 | 0.679    |
| without helmet | 0.821     | 0.917  | 0.811 | 0.655    |
| rider          | 0.915     | 0.935  | 0.952 | 0.877    |
| number plate   | 0.991     | 1.000  | 0.995 | 0.852    |

Violation detection at confidence 0.35 found 9 of the 11 true violations, with
2 false alarms and 2 misses. Precision, recall and F1 are all 0.818.

The seat-belt model reaches mAP50 ≈ 0.995 overall, with the violation class
(`Without-seatbelt`) at mAP50-95 ≈ 0.925 and recall 1.0. Datasets are small and
clean, so these numbers are indicative, not production-grade.

## Running in Google Colab

1. Open `notebooks/helmet_violation_detection.ipynb` in Google Colab.
2. Pick a GPU runtime (T4 is enough) under **Runtime > Change runtime type**.
3. Add your API keys in the Colab **Secrets** panel (🔑 in the left sidebar) —
   no keys go in the notebook:
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` (from your Kaggle account) for the dataset.
   - `ROBOFLOW_API_KEY` (only for the seat-belt dataset in Section 14).
4. Run the cells from top to bottom.

To run outside Colab, install the packages in `requirements.txt`, export the same
keys as environment variables, and change the `/content/...` paths to local ones.

## Outputs

| Path                              | Contents                          |
|-----------------------------------|-----------------------------------|
| `runs/detect/train/`              | Training curves and metrics       |
| `runs/detect/train/weights/best.pt` | Best trained weights            |
| `plate_crops/`                    | Single-plate OCR experiments      |
| `plate_results/`                  | Plate crops from all val images   |
| `final_ocr_results.csv`           | Final plate readings              |
| `violations/`                     | Annotated violation images        |
| `violations.csv`                  | Violations with their plates      |
| `annotated_output.mp4`            | Annotated video (video cell)      |

## Limitations and next steps

- The validation set has only 20 images, so the metrics above are noisy, and
  recall near 1.0 will not transfer to real CCTV footage.
- OCR on small, blurred or angled plates is still unreliable. Next steps are a
  dedicated plate-recognition model or a larger crop resolution.
- Train `yolov8s.pt` or `yolov8m.pt` for longer to improve helmet classification.
- Tracking can swap IDs in dense traffic; N-of-M confirmation can miss a violator
  seen in very few frames (a measured recall trade-off).
- One clip rarely demos both models: helmet needs two-wheeler footage, seat belt
  needs close driver/windshield views.

## Notes

API keys come from Colab Secrets — do not commit `kaggle.json`, API keys, trained
weights or generated output folders (all gitignored).

## Author

Sanskar Shekhar ([@shekharsanskar9](https://github.com/shekharsanskar9))
