# evaluate_confusion_matrix.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import string
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from inference import (
    run_yolo_detection, run_ocr_on_crops,
    YOLO_MODEL_PATH, OCR_MODEL_PATH, CLASSES, OCR_CLASSES
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROUND_TRUTH_JSON = "ground_truth.json"
OUTPUT_DIR        = "eval_outputs"
FIELD_TO_EVALUATE = None  # None = all fields, or e.g. "PAN Number"
# ──────────────────────────────────────────────────────────────────────────────

CHARACTERS = string.ascii_uppercase + string.digits + "/" + " "


def load_ground_truth(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_texts(gt_text, pred_text):
    gt_text   = gt_text.upper().strip()
    pred_text = pred_text.upper().strip()
    max_len   = max(len(gt_text), len(pred_text))
    return list(gt_text.ljust(max_len, '_')), list(pred_text.ljust(max_len, '_'))


def collect_character_pairs(ground_truth_data, yolo_model, ocr_model):
    all_gt, all_pred = [], []
    field_errors = {field: {"correct": 0, "wrong": 0} for field in OCR_CLASSES}

    for idx, sample in enumerate(ground_truth_data):
        image_path = sample["image_path"]
        labels     = sample["labels"]
        print(f"[{idx+1}/{len(ground_truth_data)}] Processing: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"  ⚠ Could not read image, skipping.")
            continue

        crops       = run_yolo_detection(img, yolo_model, CLASSES)
        ocr_results = run_ocr_on_crops(crops, ocr_model, OCR_CLASSES)

        for field in OCR_CLASSES:
            if FIELD_TO_EVALUATE and field != FIELD_TO_EVALUATE:
                continue
            if field not in labels:
                continue

            gt_text   = labels[field].upper().strip()
            pred_text = ocr_results.get(field, {}).get("text", "").upper().strip()

            gt_chars, pred_chars = align_texts(gt_text, pred_text)
            all_gt.extend(gt_chars)
            all_pred.extend(pred_chars)

            if gt_text == pred_text:
                field_errors[field]["correct"] += 1
            else:
                field_errors[field]["wrong"] += 1

    return all_gt, all_pred, field_errors


# ─── STYLED 2×2 CONFUSION MATRIX ─────────────────────────────────────────────

def plot_styled_confusion_matrix(field_errors, output_dir):
    """Styled Dog/Not-Dog style 2x2 matrix: Correct vs Incorrect at field level."""

    correct   = sum(v["correct"] for v in field_errors.values())
    incorrect = sum(v["wrong"]   for v in field_errors.values())
    total     = correct + incorrect
    accuracy  = correct / total * 100 if total > 0 else 0

    cm          = np.array([[correct, 0], [0, incorrect]])
    class_names = ["Correct", "Incorrect"]
    n           = 2

    green_cmap = LinearSegmentedColormap.from_list("green", ["#EAF3DE", "#3B6D11"])
    pink_cmap  = LinearSegmentedColormap.from_list("pink",  ["#FBEAF0", "#D4537E"])

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    for i in range(n):
        for j in range(n):
            val      = cm[i, j]
            max_diag = cm.diagonal().max()

            if i == j:
                intensity = val / max_diag if max_diag > 0 else 0
                color     = green_cmap(max(intensity, 0.15))
                text_col  = "#EAF3DE" if intensity > 0.5 else "#3B6D11"
            else:
                off_vals = cm[~np.eye(n, dtype=bool)]
                max_off  = off_vals.max() if off_vals.max() > 0 else 1
                intensity = val / max_off
                color     = pink_cmap(max(intensity, 0.08))
                text_col  = "#FBEAF0" if intensity > 0.5 else "#D4537E"

            rect = patches.FancyBboxPatch(
                (j + 0.04, n - i - 1 + 0.04),
                0.92, 0.92,
                boxstyle="round,pad=0.02",
                linewidth=0,
                facecolor=color
            )
            ax.add_patch(rect)
            ax.text(
                j + 0.5, n - i - 0.5,
                str(val),
                ha="center", va="center",
                fontsize=26, fontweight="bold",
                color=text_col
            )

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
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
        cmap=LinearSegmentedColormap.from_list("legend", ["#EAF3DE", "#3B6D11"]),
        norm=plt.Normalize(vmin=0, vmax=total)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=9, color="#5F5E5A")
    cbar.outline.set_visible(False)

    fig.text(
        0.5, 0.01,
        f"Field Accuracy: {accuracy:.1f}%   |   Total Fields: {total}   |   Errors: {incorrect}",
        ha="center", fontsize=10, color="#888780"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "styled_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"✅ Styled confusion matrix saved → {save_path}")


# ─── CHARACTER-LEVEL HEATMAP ──────────────────────────────────────────────────

def plot_confusion_matrix(all_gt, all_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    unique_chars = sorted(set(all_gt + all_pred))

    cm       = confusion_matrix(all_gt, all_pred, labels=unique_chars)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm_norm,
        xticklabels=unique_chars,
        yticklabels=unique_chars,
        cmap="Blues",
        annot=len(unique_chars) <= 20,
        fmt=".2f",
        linewidths=0.3,
        ax=ax
    )
    ax.set_xlabel("Predicted",   fontsize=13)
    ax.set_ylabel("Ground Truth", fontsize=13)
    ax.set_title("Character-Level Confusion Matrix (Normalized)", fontsize=15)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Character confusion matrix saved → {save_path}")

    confused_pairs = [
        {
            "Ground Truth": unique_chars[i],
            "Predicted":    unique_chars[j],
            "Count":        int(cm[i][j]),
            "Rate":         round(cm_norm[i][j], 3)
        }
        for i in range(len(unique_chars))
        for j in range(len(unique_chars))
        if i != j and cm[i][j] > 0
    ]

    df_confused = pd.DataFrame(confused_pairs).sort_values("Count", ascending=False)
    csv_path    = os.path.join(output_dir, "confused_pairs.csv")
    df_confused.to_csv(csv_path, index=False)
    print(f"✅ Top confused pairs saved → {csv_path}")
    print("\nTop 15 confused character pairs:")
    print(df_confused.head(15).to_string(index=False))


def print_field_accuracy(field_errors):
    print("\n─── Field-Level Accuracy ───────────────────────────")
    for field, counts in field_errors.items():
        total = counts["correct"] + counts["wrong"]
        if total == 0:
            continue
        acc = counts["correct"] / total * 100
        print(f"  {field:<15}  {counts['correct']}/{total}  ({acc:.1f}%)")
    print("────────────────────────────────────────────────────")


def main():
    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ocr_model  = load_model(OCR_MODEL_PATH, compile=False)

    print("Loading ground truth...")
    ground_truth_data = load_ground_truth(GROUND_TRUTH_JSON)
    print(f"Found {len(ground_truth_data)} samples.\n")

    all_gt, all_pred, field_errors = collect_character_pairs(
        ground_truth_data, yolo_model, ocr_model
    )

    unique_chars = sorted(set(all_gt))
    print("\n─── Character-Level Classification Report ──────────")
    print(classification_report(all_gt, all_pred, labels=unique_chars, zero_division=0))

    print_field_accuracy(field_errors)

    # Plot both matrices
    plot_confusion_matrix(all_gt, all_pred, OUTPUT_DIR)        # character heatmap
    plot_styled_confusion_matrix(field_errors, OUTPUT_DIR)     # styled 2x2


if __name__ == "__main__":
    main()