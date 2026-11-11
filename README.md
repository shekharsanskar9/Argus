# Argus

**Helmet and seat-belt violation detection with number-plate recognition.**

Argus uses YOLOv8 to find traffic violations in images and video:

- two-wheeler riders **without a helmet**, linked to the rider's **number plate**, which is read with OCR;
- car occupants **without a seat belt**, detected by a second YOLOv8 model.

Everything runs in Google Colab on a free Tesla T4 GPU. The notebooks run from top to bottom.

## Results at a glance

| Measure | Value |
|---------|-------|
| Helmet and plate detector, validation mAP50 | 0.940 (mAP50-95 0.776) |
| Violation detection at confidence 0.35 | Precision, recall and F1 all 0.909 (10 of 11 violations found, 1 false alarm, 1 missed) |
| Seat-belt detector, mAP50 | about 0.995 (repository run) |
| Rider-level ablation (V0 to V3) | Protocol implemented; results pending hand-labelled footage |

These figures come from small, clean datasets. They show the approach works, not that it is ready for production CCTV footage.

## Quick start (Colab)

1. Open [`notebooks/helmet_violation_detection.ipynb`](notebooks/helmet_violation_detection.ipynb) in Google Colab.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Add your keys in the Colab **Secrets** panel (🔑 in the left sidebar). No keys are stored in the notebooks.
   - `KAGGLE_USERNAME` and `KAGGLE_KEY`: needed to download the helmet dataset.
   - `ROBOFLOW_API_KEY`: needed only for the seat-belt dataset.
4. Run the cells from top to bottom. The video section lets you upload your own clip.

The baseline notebook, [`notebooks/helmet_basic.ipynb`](notebooks/helmet_basic.ipynb), is shorter and easier to read. It does not read Colab Secrets, so upload `kaggle.json` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` before the download cell.

### Running locally

```bash
pip install -r requirements.txt
```

Export the keys as environment variables, change the `/content/...` paths to local folders, and set `VIDEO_PATH` by hand. A CUDA GPU is strongly recommended. Apple-silicon Macs and TPUs are not supported for training.

## Repository layout

```
notebooks/
  helmet_violation_detection.ipynb   # main notebook: train, OCR, video, advanced techniques, seat belt
  helmet_basic.ipynb                 # baseline: train, detect, OCR, video
docs/
  DOCUMENTATION.md                   # full technical reference
  presentation_script_and_qa.md      # talk track and viva Q&A
  Argus_Presentation.pptx            # slide deck
samples/
  README.md                          # links to demo clips and trained weights (not in git)
requirements.txt
```

Datasets, weights, videos, CSV files and output folders are created at run time and are excluded by `.gitignore`.

## How it works

1. **Detect.** A fine-tuned YOLOv8n finds riders, helmeted heads, helmetless heads and number plates.
2. **Flag.** A rider is a violation when the centre of a `without helmet` box lies inside the rider's box.
3. **Link and read.** Each violating rider is linked to a plate, the plate is deskewed and upscaled, and EasyOCR reads it with several preprocessing variants.
4. **Track and confirm.** On video, ByteTrack gives each rider one ID. A violation is confirmed only if the rider is seen in at least 5 frames and is helmetless in at least half of them.
5. **Vote and correct.** Plate characters are combined across a rider's frames by confidence-weighted voting, then checked against the Indian plate format `SS DD L(1-3) NNNN` and a list of valid state codes.
6. **Seat belts.** A second YOLOv8 model flags `Without-seatbelt`. Both models can run on one video in a single pass.

The rider-level protocol in the main notebook reports precision, recall and F1 for the violation decision, plate exact-match rate and Character Error Rate (CER). Frame-level mAP alone does not show whether the right rider is flagged or whether the plate is read correctly.

See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for each pipeline stage, the key functions, configuration settings, outputs and troubleshooting.

## Datasets

**Helmet and plates:** [Rider with helmet / without helmet / number plate](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate) on Kaggle.

| ID | Class |
|----|-------|
| 0 | with helmet |
| 1 | without helmet |
| 2 | rider |
| 3 | number plate |

**Seat belt:** a YOLO-format dataset from Roboflow Universe with the classes `with-seatbelt` and `Without-seatbelt`. To use your own data, set `SEATBELT_DATA_YAML` to your `data.yaml`.

## Outputs

Outputs are written under `/content/` in Colab. Copy anything you want to keep to Google Drive, because `/content/` is deleted when the runtime disconnects.

| Path | Contents |
|------|----------|
| `runs/detect/train/weights/best.pt` | Helmet and plate weights |
| `violations/`, `violations.csv` | Annotated violation images and plate readings |
| `annotated_output.mp4`, `video_violations.csv` | Annotated video and violation log |
| `annotated_advanced.mp4`, `violations_advanced.csv` | Video run with the advanced techniques |
| `ablation_results.csv` | Rider-level ablation table (V0 to V3) |
| `seatbelt_best.pt` | Seat-belt weights |
| `annotated_helmet_seatbelt.mp4` | Combined helmet and seat-belt video |

## Limitations

- The validation set has only 20 images, so the metrics are noisy. Recall near 1.0 will not carry over to real CCTV footage.
- OCR is unreliable on small, blurred or angled plates.
- Tracking can swap IDs in dense traffic.
- N-of-M confirmation can miss a violator who appears in only a few frames.
- Seat-belt detection is not yet tracked, voted or evaluated at the rider level.
- One clip rarely shows both cases well: the helmet model needs two-wheeler footage, and the seat-belt model needs close views through the windshield.

## Security

API keys are read from Colab Secrets or environment variables. Never commit `kaggle.json`, API keys, trained weights or generated outputs. `.gitignore` already excludes them.

## Author

Sanskar Shekhar ([@shekharsanskar9](https://github.com/shekharsanskar9))
