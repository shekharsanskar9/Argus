# Argus: Helmet Violation Detection + Number Plate OCR

Google Colab notebooks that detect motorcycle riders without helmets and read
their number plates from traffic images.

## Notebooks

- `helmet_detection_colab.ipynb`: end-to-end pipeline. It trains a YOLOv8
  detector, flags helmet violations and reads number plates with OCR.
- `dl_training_colab.ipynb`: general deep-learning training workflow.

## Dataset

[Rider with helmet / without helmet / number plate](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate)
from Kaggle. It has four classes:

| ID | Class          |
|----|----------------|
| 0  | with helmet    |
| 1  | without helmet |
| 2  | rider          |
| 3  | number plate   |

## Pipeline

1. **Setup.** Install `ultralytics`, `kagglehub`, `easyocr` and `pyyaml`.
   Download and unzip the dataset, then write `dataset/data.yaml`.
2. **Train.** Fine-tune `yolov8n.pt` for 50 epochs at 640 px with batch size 16.
3. **Evaluate.** Report precision, recall and mAP per class.
4. **Detect violations.** A rider counts as a violation when the centre of a
   `without helmet` box falls inside the rider's box. Predicted violations are
   matched to ground truth at IoU ≥ 0.5.
5. **Crop plates.** Save every `number plate` detection from the validation
   images to `/content/plate_results/`.
6. **Deskew plates.** Estimate the plate angle with Canny edges and Hough lines,
   then rotate the crop straight. Perspective correction is also tried.
7. **OCR.** Upscale each crop 4x and make grayscale, sharpened, Otsu and
   adaptive-threshold versions. Run EasyOCR on each with an `A-Z0-9` allowlist.
   Candidates are ranked by confidence and length, with manufacturer names
   such as HONDA and YAMAHA penalised. The notebook also tries a regex score
   for the Indian plate format.
8. **Export.** Write the best reading per plate to `/content/final_ocr_results.csv`.
9. **End to end.** `analyse_image` finds each rider without a helmet, matches it
   to its number plate (the plate inside the rider's box, else the nearest one)
   and reads the plate. Annotated images go to `/content/violations/` and the
   readings to `/content/violations.csv`.
10. **Video.** `process_video` does the same per frame. It uses YOLO tracking so
    each violating rider is counted once, and keeps the best plate reading seen
    for that rider.

PaddleOCR is included as a comparison against EasyOCR.

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

## Running in Google Colab

1. Open `helmet_detection_colab.ipynb` in Google Colab.
2. Pick a GPU runtime (T4 is enough) under **Runtime > Change runtime type**.
3. Upload your Kaggle API token (`kaggle.json`) so the `kaggle` CLI can
   download the dataset.
4. Run the cells from top to bottom.

To run outside Colab, install the packages in `requirements.txt` and change the
`/content/...` paths in the notebook to local ones.

## Outputs

| Path                                  | Contents                          |
|---------------------------------------|-----------------------------------|
| `runs/detect/train/`                  | Training curves and metrics       |
| `runs/detect/train/weights/best.pt`   | Best trained weights              |
| `/content/plate_crops/`               | Single-plate OCR experiments      |
| `/content/plate_results/`             | Plate crops from all val images   |
| `/content/final_ocr_results.csv`      | Final plate readings              |
| `/content/violations/`                | Annotated violation images        |
| `/content/violations.csv`             | Violations with their plates      |
| `/content/traffic_annotated.mp4`      | Annotated video (video cell)      |

## Limitations and next steps

- The validation set has only 20 images, so the metrics above are noisy.
- OCR on small, blurred or angled plates is still unreliable. The next steps
  are a dedicated plate-recognition model or a larger crop resolution.
- Train `yolov8s.pt` or `yolov8m.pt` for longer to improve helmet
  classification.
- The video cell has not been tuned. Test it on real footage and check the
  tracker keeps rider IDs stable when riders overlap.

## Notes

Do not commit `kaggle.json`, trained weights or generated output folders.
