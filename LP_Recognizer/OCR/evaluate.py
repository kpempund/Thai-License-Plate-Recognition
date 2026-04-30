import csv
import os
import Levenshtein
from ocr_pipeline import extract_plate_text

def calculate_car(truth: str, pred: str) -> float:
    # Handle edge cases to prevent division by zero
    if not truth and not pred: return 100.0
    if not truth or not pred: return 0.0
    
    distance = Levenshtein.distance(truth, pred)
    return max(0.0, 1.0 - (distance / max(len(truth), 1))) * 100

def calculate_war(truth: str, pred: str) -> float:
    return 100.0 if truth == pred else 0.0

def evaluate_dataset(csv_path: str, image_dir: str):
    # Track everything: Number, Province, and the Combined Plate
    metrics = {
        "num_car": 0.0, "num_war": 0.0,
        "pro_car": 0.0, "pro_war": 0.0,
        "full_car": 0.0, "full_war": 0.0
    }
    evaluated_count = 0

    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_headers = {"image_filename", "true_number", "true_province"}
        missing_headers = required_headers - set(reader.fieldnames or [])
        if missing_headers:
            raise ValueError(f"CSV missing columns: {', '.join(sorted(missing_headers))}")

        for row in reader:
            image_filename = (row.get("image_filename") or "").strip()
            true_number = (row.get("true_number") or "").strip()
            true_province = (row.get("true_province") or "").strip()

            if not image_filename: continue

            image_path = os.path.join(image_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Skipping missing image: {image_path}")
                continue

            # Run OCR Pipeline
            pred = extract_plate_text(image_path)
            pred_number = (pred.get("number") or "").strip()
            pred_province = (pred.get("province") or "").strip()

            # 1. Separate Metrics
            metrics["num_car"] += calculate_car(true_number, pred_number)
            metrics["num_war"] += calculate_war(true_number, pred_number)
            metrics["pro_car"] += calculate_car(true_province, pred_province)
            metrics["pro_war"] += calculate_war(true_province, pred_province)

            # 2. Combined Metrics (The true system accuracy)
            true_combined = true_number + true_province
            pred_combined = pred_number + pred_province
            metrics["full_car"] += calculate_car(true_combined, pred_combined)
            metrics["full_war"] += calculate_war(true_combined, pred_combined)

            evaluated_count += 1

            # Fixed: Print progress INSIDE the loop
            if evaluated_count % 100 == 0:
                print(f"Processed {evaluated_count} images...")

    if evaluated_count == 0:
        print("No images were evaluated. Check your CSV paths and image directory.")
        return

    # Averages
    for key in metrics:
        metrics[key] /= evaluated_count

    print("\n=== OCR Evaluation Summary ===")
    print(f"Total Images Evaluated: {evaluated_count}")
    print(f"Number   - CAR: {metrics['num_car']:.2f}% | WAR: {metrics['num_war']:.2f}%")
    print(f"Province - CAR: {metrics['pro_car']:.2f}% | WAR: {metrics['pro_war']:.2f}%")
    print("-" * 30)
    print(f"SYSTEM OVERALL (Number + Province Combined)")
    print(f"Full Plate - CAR: {metrics['full_car']:.2f}% | WAR: {metrics['full_war']:.2f}%")

if __name__ == "__main__":
    csv_path = "labels.csv"
    image_dir = "./test_images/"

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {os.path.abspath(csv_path)}")
    elif not os.path.isdir(image_dir):
        print(f"Image directory not found: {os.path.abspath(image_dir)}")
    else:
        evaluate_dataset(csv_path, image_dir)