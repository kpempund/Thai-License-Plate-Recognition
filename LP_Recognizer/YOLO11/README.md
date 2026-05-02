# Thai License Plate Detection & OCR

Trains a YOLO11n to extract Thai character on license plates.

## Pipeline overview

| Cell | What it does |
|------|-------------|
| 1 | Download dataset from Roboflow |
| 2 | Train YOLO11n |
| 3 | Evaluate on test set |

## Trained Results

| Split | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| Val   | 0.9370  | 0.8083     | 0.9016     | 0.8993  |
| Test  | 0.9644  | 0.8914     | 0.8881     | 0.9457  |

## CAR and WAR Results

|  | CAR (Character Accuracy Rate) | WAR (Word Accuracy Rate) |
|-------|--------|-----------|
|  Number  | 92.27%  | 70.73%     |
|  Province | 85.77%  | 85.37%     |
|  Full Plate | 84.32%  | 58.54%     |


## Model

- Architecture: YOLO11n (`yolo11n.pt`)
- Input resolution: 640 × 640
- Training: 100 epochs, early stopping patience 10, batch size auto

### Augmentation settings

| Parameter | Value |
|-----------|-------|
| `perspective` | 0.0005 |
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `degrees` | 15.0 |
