# Thai License Plate Detection & OCR

Training notebook and artefacts for the plate-detection stage.

| Item | Description |
|------|-------------|
| `train_lp_detector.ipynb` | End-to-end training notebook: downloads the dataset from Roboflow, trains YOLO26s with custom augmentations (HSV, shear, perspective, mosaic, erasing), evaluates on val & test sets, and runs inference on `CarDemo.mp4` to produce `CarDemo_Inferenced.mp4`. |
| `CarDemo.mp4` | Raw input video used for qualitative demo. |
| `CarDemo_Inferenced.mp4` | Output video with bounding-box overlays from the trained detector. |
| `runs/yolo26s/` | Ultralytics run directory containing training curves, confusion matrices, and saved weights for train/val/test experiments. |

## Pipeline overview

| Cell | What it does |
|------|-------------|
| 1 | Download dataset from Roboflow |
| 2 | Train YOLO26s |
| 3 | Evaluate on validation set |
| 4 | Evaluate on test set |
| 5 | Run detection on `CarDemo.mp4` → `CarDemo_Inferenced.mp4` |

## Results

| Split | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| Val   | 0.952  | 0.706     | 0.971     | 0.882  |
| Test  | 0.949  | 0.699     | 0.984     | 0.885  |

Install with:
1. Install pytorch cuda from here: https://pytorch.org/get-started/locally/
2. Install the rest of the dependencies with ```pip install requirements.txt ```

## Model

- Architecture: YOLO26s (`yolo26s.pt`)
- Input resolution: 960 × 960
- Training: 100 epochs, early stopping patience 10, batch size 4
- Dataset: [License Plate — Roboflow](https://app.roboflow.com/twoweekfinal/license-plate-eql7j-0hwlp/1)

### Augmentation settings

| Parameter | Value |
|-----------|-------|
| `hsv_s` | 0.5 |
| `hsv_v` | 0.5 |
| `degrees` | 5.0 |
| `shear` | 3.0 |
| `perspective` | 0.0002 |
| `mosaic` | 0.5 |
| `erasing` | 0.2 |
