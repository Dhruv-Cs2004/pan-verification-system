# styled_cm.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2
import string
from tensorflow.keras.models import load_model
from ultralytics import YOLO

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from preprocess import preprocess_pan_image, is_bad_text
from postprocess import correct_ocr_errors, regex_validate
import tensorflow as tf

# ─── CONFIG ───────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH   = os.path.join(BASE_DIR, "best.pt")
OCR_MODEL_PATH    = os.path.join(BASE_DIR, "checkpoints", "final_model.keras")
GROUND_TRUTH_JSON = os.path.join(BASE_DIR, "ground_truth.json")
OUTPUT_DIR        = os.path.join(BASE_DIR, "eval_outputs")

CLASSES     = ['DOB', 'Father Name', 'Name', 'PAN', 'PAN Number', 'Photo', 'QR', 'Signature']
OCR_CLASSES = ["DOB", "PAN Number", "Name", "Father Name"]
IMG_HEIGHT, IMG_WIDTH = 64, 256
CHARACTERS  = string.ascii_uppercase + string.digits + "/" + " "
NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
# ──────────────────────────────────────────────────────────────────────────────


def preprocess_for_ocr(image):
    processed = preprocess_pan_image(image)
    if processed is None:
        return None
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    processed = cv2.resize(processed, (IMG_WIDTH, IMG_HEIGHT))
    return processed.astype('float32') / 255.0


def decode_prediction(prediction):
    input_length = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded, _   = tf.keras.backend.ctc_decode(prediction, input_length, greedy=True)
    decoded_tensor = decoded[0]
    if isinstance(decoded_tensor, tf.SparseTensor):
        dense = tf.sparse.to_dense(decoded_tensor).numpy()
    else:
        dense = decoded_tensor.numpy() if hasattr(decoded_tensor, "numpy") else np.array(decoded_tensor)
    return ["".join([NUM_TO_CHAR[i] for i in seq if i != -1 and i in NUM_TO_CHAR]) for seq in dense]


def run_yolo_detection(image, yolo_model):
    results  = yolo_model(image)[0]
    boxes    = results.boxes.xyxy.cpu().numpy()
    confs    = results.boxes.conf.cpu().numpy()
    classes  = results.boxes.cls.cpu().numpy()
    best, crops = {}, {}
    for idx, cls in enumerate(classes):
        cls = int(cls)
        if cls not in best or confs[idx] > best[cls][1]:
            best[cls] = (boxes[idx], confs[idx])
    for cls, (box, _) in best.items():
        x1, y1, x2, y2 = map(int, box)
        crops[CLASSES[cls]] = image[y1:y2, x1:x2]
    return crops


def ocr_field(crop, ocr_model, field_type):
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    proc = preprocess_for_ocr(crop)
    if proc is None:
        return ""
    text0 = decode_prediction(ocr_model.predict(np.expand_dims(proc, 0), verbose=0))[0]
    if not is_bad_text(text0, expected_format=field_type):
        return text0
    proc180 = preprocess_for_ocr(cv2.rotate(crop, cv2.ROTATE_180))
    if proc180 is None:
        return text0
    text180 = decode_prediction(ocr_model.predict(np.expand_dims(proc180, 0), verbose=0))[0]
    if not is_bad_text(text180, expected_format=field_type):
        return text180
    return text0 if len(text0) >= len(text180) else text180


def run_ocr(crops, ocr_model):
    results = {}
    for field in OCR_CLASSES:
        if field in crops:
            ft   = field.lower().replace(" ", "_")
            text = ocr_field(crops[field], ocr_model, ft)
            results[field] = correct_ocr_errors(text, ft)
        else:
            results[field] = ""
    return results


# ─── STYLED 2×2 PLOT ──────────────────────────────────────────────────────────

def plot_styled_cm(correct, incorrect, output_dir):
    total    = correct + incorrect
    accuracy = correct / total * 100 if total > 0 else 0

    cm          = np.array([[correct, 0], [0, incorrect]])
    class_names = ["Correct", "Incorrect"]
    n           = 2

    green_cmap = LinearSegmentedColormap.from_list("g", ["#EAF3DE", "#3B6D11"])
    pink_cmap  = LinearSegmentedColormap.from_list("p", ["#FBEAF0", "#D4537E"])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if i == j:
                intensity = val / max(cm.diagonal().max(), 1)
                color     = green_cmap(max(intensity, 0.15))
                text_col  = "#EAF3DE" if intensity > 0.5 else "#3B6D11"
            else:
                off_max   = max(cm[~np.eye(n, dtype=bool)].max(), 1)
                intensity = val / off_max
                color     = pink_cmap(max(intensity, 0.08))
                text_col  = "#FBEAF0" if intensity > 0.5 else "#D4537E"

            ax.add_patch(patches.FancyBboxPatch(
                (j + 0.04, n - i - 1 + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02", linewidth=0, facecolor=color
            ))
            ax.text(j + 0.5, n - i - 0.5, str(val),
                    ha="center", va="center",
                    fontsize=26, fontweight="bold", color=text_col)

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(class_names, fontsize=11, color="#5F5E5A")
    ax.set_yticklabels(list(reversed(class_names)), fontsize=11, color="#5F5E5A")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Prediction", fontsize=12, color="#5F5E5A", labelpad=10)
    ax.set_ylabel("Actual",     fontsize=12, color="#5F5E5A", labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=13, color="#2C2C2A", pad=14, fontweight="normal")

    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("l", ["#EAF3DE", "#3B6D11"]),
        norm=plt.Normalize(vmin=0, vmax=total)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=9); cbar.outline.set_visible(False)

    fig.text(0.5, 0.01,
             f"Field Accuracy: {accuracy:.1f}%   |   Total: {total}   |   Errors: {incorrect}",
             ha="center", fontsize=10, color="#888780")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "styled_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n✅ Styled confusion matrix saved → {save_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ocr_model  = load_model(OCR_MODEL_PATH, compile=False)

    print("Loading ground truth...")
    with open(GROUND_TRUTH_JSON, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    print(f"Found {len(ground_truth)} samples.\n")

    correct = incorrect = 0
    field_results = {f: {"correct": 0, "wrong": 0} for f in OCR_CLASSES}

    for idx, sample in enumerate(ground_truth):
        img = cv2.imread(sample["image_path"])
        if img is None:
            print(f"  ⚠ Skipping: {sample['image_path']}")
            continue
        print(f"[{idx+1}/{len(ground_truth)}] {sample['image_path']}")

        crops   = run_yolo_detection(img, yolo_model)
        results = run_ocr(crops, ocr_model)

        for field, gt_text in sample["labels"].items():
            if field not in OCR_CLASSES:
                continue
            pred_text = results.get(field, "").upper().strip()
            gt_text   = gt_text.upper().strip()

            if gt_text == pred_text:
                correct += 1
                field_results[field]["correct"] += 1
            else:
                incorrect += 1
                field_results[field]["wrong"] += 1
                print(f"  ✗ {field}: GT='{gt_text}' | PRED='{pred_text}'")

    # Print field accuracy
    print("\n─── Field-Level Accuracy ───────────────────────────")
    for field, counts in field_results.items():
        total = counts["correct"] + counts["wrong"]
        if total == 0: continue
        print(f"  {field:<15}  {counts['correct']}/{total}  ({counts['correct']/total*100:.1f}%)")
    print("────────────────────────────────────────────────────")

    # Plot styled matrix
    plot_styled_cm(correct, incorrect, OUTPUT_DIR)


if __name__ == "__main__":
    main()