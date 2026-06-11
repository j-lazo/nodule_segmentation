#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# Basic I/O
# ============================================================

def load_history(history_json_path):
    """
    Load training history.json into a pandas DataFrame.
    Expected format: list of dicts, one per epoch.
    """
    history_json_path = Path(history_json_path)
    with open(history_json_path, "r") as f:
        history = json.load(f)

    df = pd.DataFrame(history)
    return df


def load_test_metrics(csv_path):
    """
    Load test_metrics.csv into a DataFrame.
    """
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path)


def load_volume(path):
    """
    Load a medical volume and return:
      - numpy array [D,H,W]
      - original SimpleITK image
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


# ============================================================
# Training history analysis
# ============================================================

def summarize_history(df):
    """
    Print and return summary statistics from history DataFrame.
    """
    out = {}

    if "epoch" in df.columns:
        out["num_epochs"] = int(df["epoch"].max())
    else:
        out["num_epochs"] = len(df)

    if "train_loss" in df.columns:
        out["best_train_loss"] = float(df["train_loss"].min())
        out["final_train_loss"] = float(df["train_loss"].iloc[-1])

    if "val_loss" in df.columns:
        out["best_val_loss"] = float(df["val_loss"].min())
        out["final_val_loss"] = float(df["val_loss"].iloc[-1])

    if "train_dice" in df.columns:
        out["best_train_dice"] = float(df["train_dice"].max())
        out["final_train_dice"] = float(df["train_dice"].iloc[-1])

    if "val_dice" in df.columns:
        best_idx = df["val_dice"].idxmax()
        out["best_val_dice"] = float(df.loc[best_idx, "val_dice"])
        out["best_val_epoch"] = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else int(best_idx + 1)
        out["final_val_dice"] = float(df["val_dice"].iloc[-1])

    if "lr" in df.columns:
        out["initial_lr"] = float(df["lr"].iloc[0])
        out["final_lr"] = float(df["lr"].iloc[-1])

    print("\n=== Training History Summary ===")
    for k, v in out.items():
        print(f"{k}: {v}")

    return out


def plot_training_history(df, save_path=None, show=True):
    """
    Plot loss and dice curves from history DataFrame.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    if "train_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Dice plot
    if "train_dice" in df.columns:
        axes[1].plot(df["epoch"], df["train_dice"], label="train_dice")
    if "val_dice" in df.columns:
        axes[1].plot(df["epoch"], df["val_dice"], label="val_dice")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def detect_overfitting_epoch(df: pd.DataFrame, patience: int = 5) -> Optional[int]:
    """
    Very simple heuristic:
    returns the first epoch after which val_dice does not improve for `patience` epochs
    while train_dice keeps increasing.
    """
    if "train_dice" not in df.columns or "val_dice" not in df.columns:
        return None

    best_val = -np.inf
    epochs_since_best = 0

    for i in range(len(df)):
        val_d = df.iloc[i]["val_dice"]
        if val_d > best_val:
            best_val = val_d
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epochs_since_best >= patience:
            return int(df.iloc[i]["epoch"])

    return None


# ============================================================
# Test metrics analysis
# ============================================================

def summarize_test_metrics(df: pd.DataFrame) -> Dict:
    """
    Summarize test metrics CSV.
    Ignores special rows like __mean__ if present.
    """
    df_cases = df[df["case_id"] != "__mean__"].copy() if "case_id" in df.columns else df.copy()

    out = {}
    if "dice" in df_cases.columns:
        out["mean_dice"] = float(df_cases["dice"].mean())
        out["std_dice"] = float(df_cases["dice"].std())
        out["min_dice"] = float(df_cases["dice"].min())
        out["max_dice"] = float(df_cases["dice"].max())

    if "iou" in df_cases.columns:
        out["mean_iou"] = float(df_cases["iou"].mean())
        out["std_iou"] = float(df_cases["iou"].std())

    print("\n=== Test Metrics Summary ===")
    for k, v in out.items():
        print(f"{k}: {v}")

    return out


def plot_test_metric_histograms(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    show: bool = True,
):
    """
    Plot histograms for Dice and IoU.
    """
    df_cases = df[df["case_id"] != "__mean__"].copy() if "case_id" in df.columns else df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "dice" in df_cases.columns:
        axes[0].hist(df_cases["dice"].dropna(), bins=20)
        axes[0].set_title("Dice Distribution")
        axes[0].set_xlabel("Dice")
        axes[0].set_ylabel("Count")
        axes[0].grid(True)

    if "iou" in df_cases.columns:
        axes[1].hist(df_cases["iou"].dropna(), bins=20)
        axes[1].set_title("IoU Distribution")
        axes[1].set_xlabel("IoU")
        axes[1].set_ylabel("Count")
        axes[1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def get_best_and_worst_cases(
    df: pd.DataFrame,
    metric: str = "dice",
    top_k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return best and worst cases according to a metric.
    """
    df_cases = df[df["case_id"] != "__mean__"].copy() if "case_id" in df.columns else df.copy()
    df_cases = df_cases.sort_values(metric, ascending=False)

    best = df_cases.head(top_k).reset_index(drop=True)
    worst = df_cases.tail(top_k).reset_index(drop=True)

    return best, worst


# ============================================================
# Prediction volume analysis
# ============================================================

def list_prediction_files(predictions_dir: str | Path, suffix: str = ".nii.gz") -> List[Path]:
    predictions_dir = Path(predictions_dir)
    return sorted(predictions_dir.glob(f"*{suffix}"))


def compute_binary_segmentation_stats(volume: np.ndarray) -> Dict:
    """
    Compute simple stats for a predicted binary mask.
    """
    volume = (volume > 0).astype(np.uint8)

    voxels = int(volume.sum())

    if voxels > 0:
        coords = np.argwhere(volume > 0)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)

        bbox_shape = (
            int(zmax - zmin + 1),
            int(ymax - ymin + 1),
            int(xmax - xmin + 1),
        )
    else:
        bbox_shape = (0, 0, 0)

    return {
        "foreground_voxels": voxels,
        "bbox_depth": bbox_shape[0],
        "bbox_height": bbox_shape[1],
        "bbox_width": bbox_shape[2],
        "is_empty": int(voxels == 0),
    }


def analyze_predictions_folder(predictions_dir: str | Path) -> pd.DataFrame:
    """
    Analyze all predicted volumes in predictions folder.
    """
    rows = []
    pred_files = list_prediction_files(predictions_dir)

    for p in pred_files:
        arr, _ = load_volume(p)
        stats = compute_binary_segmentation_stats(arr)
        stats["case_id"] = p.name.replace(".nii.gz", "")
        rows.append(stats)

    df = pd.DataFrame(rows)

    print("\n=== Prediction Folder Summary ===")
    print(f"num_predictions: {len(df)}")
    if len(df) > 0 and "foreground_voxels" in df.columns:
        print(f"mean_foreground_voxels: {df['foreground_voxels'].mean():.2f}")
        print(f"empty_predictions: {df['is_empty'].sum()}")

    return df


# ============================================================
# Qualitative visualization
# ============================================================

def find_best_slice(mask_3d: np.ndarray) -> int:
    """
    Return slice index with largest mask area.
    """
    areas = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def show_prediction_overlay(
    image_path: str | Path,
    pred_path: str | Path,
    gt_path: Optional[str | Path] = None,
    slice_idx: Optional[int] = None,
    alpha_pred: float = 0.4,
    alpha_gt: float = 0.4,
):
    """
    Show one slice with prediction overlay, and GT if provided.
    """
    image, _ = load_volume(image_path)
    pred, _ = load_volume(pred_path)
    pred = (pred > 0).astype(np.uint8)

    gt = None
    if gt_path is not None:
        gt, _ = load_volume(gt_path)
        gt = (gt > 0).astype(np.uint8)

    if slice_idx is None:
        if gt is not None and gt.sum() > 0:
            slice_idx = find_best_slice(gt)
        elif pred.sum() > 0:
            slice_idx = find_best_slice(pred)
        else:
            slice_idx = image.shape[0] // 2

    img2d = image[slice_idx]
    pred2d = pred[slice_idx]

    plt.figure(figsize=(6, 6))
    plt.imshow(img2d, cmap="gray")

    if gt is not None:
        gt2d = gt[slice_idx]
        gt_overlay = np.ma.masked_where(gt2d == 0, gt2d)
        plt.imshow(gt_overlay, alpha=alpha_gt)

    pred_overlay = np.ma.masked_where(pred2d == 0, pred2d)
    plt.imshow(pred_overlay, alpha=alpha_pred)

    title = f"Slice {slice_idx}"
    if gt_path is not None:
        title += " | GT + Prediction"
    else:
        title += " | Prediction"
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# End-to-end experiment analysis
# ============================================================

def analyze_experiment_folder(experiment_dir, save_plots=True,):
    """
    Run a basic analysis over:
      - history.json
      - test_metrics.csv
      - predictions/
    """
    experiment_dir = Path(experiment_dir)
    results = {}

    history_path = experiment_dir / "history.json"
    test_csv_path = experiment_dir / "test_metrics.csv"
    predictions_dir = experiment_dir / "predictions"

    if history_path.exists():
        hist_df = load_history(history_path)
        results["history_summary"] = summarize_history(hist_df)

        overfit_epoch = detect_overfitting_epoch(hist_df, patience=5)
        results["overfitting_epoch_estimate"] = overfit_epoch
        print(f"overfitting_epoch_estimate: {overfit_epoch}")

        plot_path = experiment_dir / "training_curves.png" if save_plots else None
        plot_training_history(hist_df, save_path=plot_path, show=not save_plots)

    if test_csv_path.exists():
        test_df = load_test_metrics(test_csv_path)
        results["test_summary"] = summarize_test_metrics(test_df)

        hist_path = experiment_dir / "test_metric_histograms.png" if save_plots else None
        plot_test_metric_histograms(test_df, save_path=hist_path, show=not save_plots)

        best_cases, worst_cases = get_best_and_worst_cases(test_df, metric="dice", top_k=10)
        results["best_cases"] = best_cases
        results["worst_cases"] = worst_cases

        print("\n=== Best Cases ===")
        print(best_cases)

        print("\n=== Worst Cases ===")
        print(worst_cases)

    if predictions_dir.exists():
        pred_df = analyze_predictions_folder(predictions_dir)
        results["prediction_stats"] = pred_df

        pred_csv = experiment_dir / "prediction_volume_stats.csv"
        pred_df.to_csv(pred_csv, index=False)
        print(f"\nSaved prediction stats CSV to: {pred_csv}")

    return results


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    experiment_dir = "results_3d_unet"
    analyze_experiment_folder(experiment_dir, save_plots=True)