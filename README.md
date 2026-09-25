# Helmet Violation Detection

Google Colab notebooks for detecting riders without helmets and reading vehicle
number plates from traffic images.

## Notebooks

- `helmet_detection_colab.ipynb` - End-to-end helmet violation detection and
  number-plate OCR pipeline.
- `dl_training_colab.ipynb` - Deep-learning training workflow.

## Pipeline

1. Download the Kaggle rider helmet dataset.
2. Repair the dataset paths in `data.yaml`.
3. Train a YOLOv8 object-detection model.
4. Validate the model and inspect predictions.
5. Detect riders without helmets.
6. Crop number plates and read them with EasyOCR.

## Running in Google Colab

1. Open a notebook in Google Colab.
2. Enable a GPU runtime, preferably a T4: **Runtime > Change runtime type**.
3. Run the setup and package-installation cells.
4. Create a Kaggle API token and upload `kaggle.json` when prompted.
5. Run the cells from top to bottom.

The notebook installs `ultralytics`, `kagglehub`, `easyocr`, and `pyyaml`.
Training with the default YOLOv8n configuration takes about 15-25 minutes on a
T4 GPU.

## Outputs

- Training results are written under `runs/helmet/`.
- The best trained weights are saved as `best.pt` in the training output.
- Custom images can be uploaded for one-off predictions near the end of the
  notebook.

## Improving Results

- Increase training from 30 to 100 epochs.
- Try `yolov8s.pt` or `yolov8m.pt` for a larger model.
- Use PaddleOCR when better recognition on angled plates is needed.
- For video, process frames with OpenCV and write annotated frames to a video
  file.

## Notes

- Do not commit `kaggle.json`, trained weights, or generated output folders.
- Dataset class names can vary. Check the printed model class names if the
  notebook does not identify helmet or rider roles correctly.