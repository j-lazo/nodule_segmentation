#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a 3D U-Net for CT nodule segmentation.

Expected folder structure
-------------------------
images_dir/
    case_001.mha
    case_002.mha
    ...

masks_dir/
    case_001.nii.gz
    case_002.nii.gz
    ...

The code matches images and masks by case id:
- image id = filename without ".mha"
- mask id  = filename without ".nii.gz"

Example
-------
python train_3d_unet_luna.py \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --output_dir ./results_3d_unet \
    --epochs 100 \
    --batch_size 2 \
    --patch_size 128 128 128
"""

from __future__ import annotations

import os
import math
import random
import numpy as np
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


from models import unet_3d 
from utils import results_analysis as ra
from utils import file_processing as fp
from utils import losses_and_metrics as lm
from utils import image_processing_functions as ip
from utils import unet_segmentation_data_loaders as dl
from monai.inferers import sliding_window_inference



# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  
# ============================================================
# Train / validate
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Train {epoch:03d}/{epochs:03d}", leave=False, dynamic_ncols=True)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        batch_dice = lm.dice_coefficient(logits.detach(), masks)

        running_loss += loss.item() * bs
        running_dice += batch_dice * bs
        n += bs

        avg_loss = running_loss / n
        avg_dice = running_dice / n
        lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "dice": f"{avg_dice:.4f}",
            "lr": f"{lr:.2e}"
                        })

    return running_loss / n, running_dice / n


def pad_to_factor_3d(
    volume: np.ndarray,
    factor: int = 16,
    mode: str = "constant",
    constant_values: float = 0,
):
    """
    Pad [D,H,W] so each dimension is divisible by factor.
    Returns padded volume and padding info.
    """
    D, H, W = volume.shape

    pad_d = (factor - D % factor) % factor
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor

    pad_before = (pad_d // 2, pad_h // 2, pad_w // 2)
    pad_after = (
        pad_d - pad_before[0],
        pad_h - pad_before[1],
        pad_w - pad_before[2],
    )

    padded = np.pad(
        volume,
        (
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]),
        ),
        mode=mode,
        constant_values=constant_values,
    )

    pad_info = {
        "pad_before": pad_before,
        "pad_after": pad_after,
        "original_shape": (D, H, W),
    }

    return padded, pad_info


def unpad_3d(volume, pad_info):
    """
    Remove padding from [D,H,W].
    """
    D, H, W = pad_info["original_shape"]
    pb = pad_info["pad_before"]

    z0, y0, x0 = pb
    return volume[z0:z0 + D, y0:y0 + H, x0:x0 + W]


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch, epochs):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Val   {epoch:03d}/{epochs:03d}", leave=False, dynamic_ncols=True)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        bs = images.size(0)
        batch_dice = lm.dice_coefficient(logits, masks)

        running_loss += loss.item() * bs
        running_dice += batch_dice * bs
        n += bs

        avg_loss = running_loss / n
        avg_dice = running_dice / n

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "dice": f"{avg_dice:.4f}",
        })

    return running_loss / n, running_dice / n


@torch.no_grad()
def validate_one_epoch_sliding_window(model, loader, criterion, device, epoch,epochs, roi_size, sw_batch_size=1, overlap=0.5, threshold=0.5,):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"ValSW {epoch:03d}/{epochs:03d}", leave=False, dynamic_ncols=True)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)  # [1,1,D,H,W]
        masks = batch["mask"].to(device, non_blocking=True)    # [1,1,D,H,W]

        logits = sliding_window_inference(inputs=images, roi_size=roi_size, sw_batch_size=sw_batch_size, 
                                          predictor=model, overlap=overlap, mode="gaussian", 
                                          padding_mode="constant", cval=0.0, progress=False,)

        loss = criterion(logits, masks)
        batch_dice = lm.dice_coefficient(logits, masks, threshold=threshold)

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_dice += batch_dice * bs
        n += bs

        avg_loss = running_loss / n
        avg_dice = running_dice / n

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "dice": f"{avg_dice:.4f}",
        })

    return running_loss / n, running_dice / n


@torch.no_grad()
def evaluate_on_test_set(model, loader, device, output_dir, save_predictions, threshold=0.5,):
    
    model.eval()

    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)   # [1,1,D,H,W]
        masks = batch["mask"].to(device, non_blocking=True)     # [1,1,D,H,W]
        case_id = batch["case_id"][0]

        logits = model(images)
        probs = torch.sigmoid(logits)

        metrics = lm.compute_case_metrics_from_probs(probs, masks, threshold=threshold)

        pred_bin = (probs > threshold).float()
        pred_np = pred_bin[0, 0].cpu().numpy().astype(np.uint8)

        row = {
            "case_id": case_id,
            "dice": float(metrics["dice"][0]),
            "iou": float(metrics["iou"][0]),
            "pred_voxels": int(metrics["pred_voxels"][0]),
            "target_voxels": int(metrics["target_voxels"][0]),
        }
        rows.append(row)

        if save_predictions:
            # recover original reference image path from dataset sample
            dataset = loader.dataset
            sample_map = {s["case_id"]: s for s in dataset.samples}
            ref_image_path = sample_map[case_id]["image"]

            out_path = pred_dir / f"{case_id}.nii.gz"
            ip.save_prediction_nifti(pred_np, ref_image_path, str(out_path))

    # summary row
    if len(rows) > 0:
        mean_dice = float(np.mean([r["dice"] for r in rows]))
        mean_iou = float(np.mean([r["iou"] for r in rows]))
    else:
        mean_dice = float("nan")
        mean_iou = float("nan")

    summary = [{
        "case_id": "__mean__",
        "dice": mean_dice,
        "iou": mean_iou,
        "pred_voxels": "",
        "target_voxels": "",
    }]

    fp.save_csv(rows + summary, output_dir / "test_metrics.csv")

    return rows, {"mean_dice": mean_dice, "mean_iou": mean_iou}



@torch.no_grad()
def evaluate_on_test_set_sliding_window(model, loader, device, output_dir, roi_size=(128, 128, 128), sw_batch_size=2,
                                        overlap=0.5, save_predictions=True, threshold=0.5,):
    
    model.eval()

    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for batch in tqdm(loader, desc="Test SW", dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)  # [1,1,D,H,W]
        masks = batch["mask"].to(device, non_blocking=True)    # [1,1,D,H,W]
        case_id = batch["case_id"][0]
        ref_image_path = batch["image_path"][0]

        logits = sliding_window_inference(inputs=images, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap, mode="gaussian",
                                          padding_mode="constant", cval=0.0, progress=False,)

        probs = torch.sigmoid(logits)
        metrics = lm.compute_case_metrics_from_probs(probs, masks, threshold=threshold)

        pred_bin = (probs > threshold).float()
        pred_np = pred_bin[0, 0].cpu().numpy().astype(np.uint8)

        rows.append({
            "case_id": case_id,
            "dice": float(metrics["dice"][0]),
            "iou": float(metrics["iou"][0]),
            "pred_voxels": int(metrics["pred_voxels"][0]),
            "target_voxels": int(metrics["target_voxels"][0]),})

        if save_predictions:
            out_path = pred_dir / f"{case_id}.nii.gz"
            ip.save_prediction_nifti(pred_np, ref_image_path, str(out_path))

    mean_dice = float(np.mean([r["dice"] for r in rows])) if rows else float("nan")
    mean_iou = float(np.mean([r["iou"] for r in rows])) if rows else float("nan")

    summary = [{
        "case_id": "__mean__",
        "dice": mean_dice,
        "iou": mean_iou,
        "pred_voxels": "",
        "target_voxels": "",
    }]
    fp.save_csv(rows + summary, output_dir / "test_metrics.csv")

    return rows, {"mean_dice": mean_dice, "mean_iou": mean_iou}


@torch.no_grad()
def predict_roi_and_paste(model, image, boxes, device, factor=16, threshold=0.5,):
    """
    image: normalized full image [D,H,W]
    boxes: list of (z0,y0,x0,z1,y1,x1)
    returns full prediction [D,H,W]
    """
    model.eval()

    D, H, W = image.shape
    full_prob = np.zeros((D, H, W), dtype=np.float32)

    for box in boxes:
        z0, y0, x0, z1, y1, x1 = box

        crop = image[z0:z1, y0:y1, x0:x1]
        crop_padded, pad_info = pad_to_factor_3d(crop, factor=factor, mode="constant", constant_values=0,)
        x = torch.from_numpy(crop_padded).float().unsqueeze(0).unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        probs = unpad_3d(probs, pad_info)

        # combine overlapping boxes by max probability
        full_prob[z0:z1, y0:y1, x0:x1] = np.maximum(full_prob[z0:z1, y0:y1, x0:x1], probs,)

    full_pred = (full_prob > threshold).astype(np.uint8)
    return full_pred, full_prob


@torch.no_grad()
def evaluate_nodulenet_style_roi_test(model, test_samples, device, output_dir, roi_margin=32, factor=16, 
                                      threshold = 0.5, save_predictions = True,):

    """
    NoduleNet/MANet-like segmentation test:
    candidate box -> ROI crop -> mask prediction -> paste back to full volume.

    By default candidate boxes are oracle boxes from GT mask.
    """
    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for item in tqdm(test_samples, desc="ROI test", dynamic_ncols=True):
        case_id = item["case_id"]

        image, _ = ip.load_mha(item["image"])
        gt, _ = ip.load_nii_mask(item["mask"])

        original_shape = image.shape
        image_norm = ip.clip_and_scale_ct(image)
        
        # this is the thing should be changed and itterate over the whole image
        boxes = ip.get_oracle_candidate_boxes_from_mask(gt, margin=roi_margin, factor=factor,)

        if len(boxes) == 0:
            pred = np.zeros_like(gt, dtype=np.uint8)
        else:
            pred, prob = predict_roi_and_paste(model=model, image=image_norm, boxes=boxes, device=device, factor=factor, threshold=threshold)
                
        assert pred.shape == original_shape, (
            f"Prediction shape mismatch for {case_id}: "
            f"pred={pred.shape}, image={original_shape}")

        dsc = lm.dice_numpy(pred, gt)
        iou = lm.iou_numpy(pred, gt)

        row = {
            "case_id": case_id,
            "dice": dsc,
            "iou": iou,
            "num_boxes": len(boxes),
            "pred_voxels": int(pred.sum()),
            "target_voxels": int(gt.sum()),
            "original_D": int(original_shape[0]),
            "original_H": int(original_shape[1]),
            "original_W": int(original_shape[2]),
        }
        rows.append(row)

        if save_predictions:
            out_path = pred_dir / f"{case_id}.nii.gz"
            ip.save_prediction_like_reference(pred, item["image"], out_path)

    mean_dice = float(np.mean([r["dice"] for r in rows])) if rows else float("nan")
    mean_iou = float(np.mean([r["iou"] for r in rows])) if rows else float("nan")

    rows.append({
        "case_id": "__mean__",
        "dice": mean_dice,
        "iou": mean_iou,
        "num_boxes": "",
        "pred_voxels": "",
        "target_voxels": "",
        "original_D": "",
        "original_H": "",
        "original_W": "",
    })

    fp.save_csv(rows, output_dir / "test_metrics_roi.csv")

    return rows, {"mean_dice": mean_dice, "mean_iou": mean_iou}
    
# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_loader", type=str, default="Luna_25", choices=["Luna_16", "Luna_25"])
    parser.add_argument("--path_link_file", type=str, default=None, help="Path to the .csv file linking image and mask ids for paired loading (required if --dataset_loader is Luna_16).")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=5)
    
    parser.add_argument("--make_test_split", action="store_true",
                    help="If set and no external test set is given, create a test split from the main dataset.")

    parser.add_argument("--test_ratio", type=float, default=0.1,
                    help="Fraction of full dataset to reserve for test when --make_test_split is used.")
    
    parser.add_argument("--test_images_dir", type=str, default=None)
    parser.add_argument("--test_masks_dir", type=str, default=None)
    parser.add_argument("--save_test_predictions", action="store_true")
    parser.add_argument("--evaluate_test_on_sliding_window", action="store_true")

    #parser.add_argument("--val_mode", type=str, default="patch", choices=["patch", "sliding_window"], help="Validation mode: patch-based or full-volume sliding-window.")
    parser.add_argument("--val_sw_batch_size", type=int, default=1, help="Number of sliding-window patches processed per forward pass during validation.")
    parser.add_argument("--val_sw_overlap", type=float, default=0.5, help="Overlap fraction for sliding-window validation.")

    parser.add_argument("--train_positive_crop_prob", type=float, default=0.9, help="Croping probability in the training data loader of boxels with nodules.")
    parser.add_argument("--val_positive_crop_prob", type=float, default=0.7, help="Croping probability in the validation data loader of boxels with nodules .")

    
    parser.add_argument("--val_sw_every", type=int, default=0, help="Run sliding-window validation every N epochs. 0 disables it.")
    parser.add_argument("--save_best_on", type=str, default="patch", choices=["patch", "sliding_window"], help="Which validation metric to use for saving the best checkpoint.")
    
    return parser.parse_args()


def main():

    args = parse_args()
    set_seed(args.seed)
    
    exp_dir = fp.make_experiment_dir(args, root_dir=os.path.join(os.getcwd(), "results"))
    ckpt_dir = exp_dir / "checkpoints"
    fp.ensure_dir(exp_dir)
    fp.ensure_dir(ckpt_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment directory: {exp_dir}")

    factor = 2 ** (args.depth - 1)
    patch_size = tuple(args.patch_size)
    internal_test_ratio = args.test_ratio if (args.make_test_split and args.test_images_dir is None and args.test_masks_dir is None) else 0.0
    internal_test_loader = None
    
    #train_loader, val_loader, internal_test_loader, train_samples, val_samples, internal_test_samples = dl.create_dataloaders(images_dir=args.images_dir,
    #                                                                                                                          masks_dir=args.masks_dir,
    #                                                                                                                          patch_size=patch_size,
    #                                                                                                                          batch_size=args.batch_size,
    #                                                                                                                          num_workers=args.num_workers,
    #                                                                                                                          val_ratio=args.val_ratio,
    #                                                                                                                          test_ratio=internal_test_ratio,
    #                                                                                                                          seed=args.seed,
    #                                                                                                                          val_mode=args.val_mode,)

    if args.dataset_loader == 'Luna_25':
    
        train_loader, val_loader, val_loader_sw, internal_test_loader, train_samples, val_samples, internal_test_samples = dl.create_dataloaders_luna25(images_dir=args.images_dir,
                                                                                                                                                    masks_dir=args.masks_dir,
                                                                                                                                                    patch_size=patch_size,
                                                                                                                                                    batch_size=args.batch_size,
                                                                                                                                                    num_workers=args.num_workers,
                                                                                                                                                    val_ratio=args.val_ratio,
                                                                                                                                                    test_ratio=internal_test_ratio,
                                                                                                                                                    seed=args.seed,
                                                                                                                                                    train_positive_crop_prob=args.train_positive_crop_prob, 
                                                                                                                                                    val_positive_crop_prob=args.val_positive_crop_prob) 
        print(f"Train cases: {len(train_samples)}")
        print(f"Val cases:   {len(val_samples)}")
        print(f"Internal test cases: {len(internal_test_samples)}")
        fp.save_json(internal_test_samples, exp_dir / "test_split.json")
        
    elif args.dataset_loader == 'Luna_16':
        train_loader, val_loader, val_loader_sw, test_loader, train_samples, val_samples, test_samples = dl.create_dataloaders_luna16(path_volumes=args.images_dir, 
                                                                                                          path_masks=args.masks_dir, 
                                                                                                          path_ids_lin_file=args.path_link_file, 
                                                                                                          patch_size=patch_size, 
                                                                                                          batch_size=args.batch_size, 
                                                                                                          num_workers=args.num_workers,
                                                                                                          test_ratio=internal_test_ratio,
                                                                                                          train_ratio= args.train_ratio,
                                                                                                          val_ratio=args.val_ratio,
                                                                                                          seed=args.seed,
                                                                                                          train_positive_crop_prob=args.train_positive_crop_prob, 
                                                                                                          val_positive_crop_prob=args.val_positive_crop_prob
                                                                                                          )
        print(f"Train cases: {len(train_samples)}")
        print(f"Val cases:   {len(val_samples)}")
        print(f"Internal test cases: {len(test_samples)}")
        fp.save_json(test_samples, exp_dir / "test_split.json")
  

    args_dict = vars(args).copy()
    args_dict["resolved_output_dir"] = str(exp_dir)
    

    fp.save_json(train_samples, exp_dir / "train_split.json")
    fp.save_json(val_samples, exp_dir / "val_split.json")
    fp.save_json(args_dict, exp_dir / "config.json")

    model = unet_3d.UNet3D(in_channels=1, out_channels=1, depth=args.depth, base_ch=args.base_channels, trilinear=True).to(device)

    criterion = lm.BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    history = []
    metric_for_best = None

    
    
    for epoch in range(1, args.epochs + 1):

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        
        # Fast validation every epoch
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, epoch, args.epochs)
        
        # Optional slow full-volume validation every N epochs
        val_sw_loss = None
        val_sw_dice = None
        run_sw_val = args.val_sw_every > 0 and (epoch % args.val_sw_every == 0)
        
        if run_sw_val:
            val_sw_loss, val_sw_dice = validate_one_epoch_sliding_window(model=model, loader=val_loader_sw, criterion=criterion, device=device, epoch=epoch, epochs=args.epochs, roi_size=patch_size,
                                                                         sw_batch_size=args.val_sw_batch_size, overlap=args.val_sw_overlap, threshold=0.5,)
        
        if args.save_best_on == "patch":
                metric_for_best = val_dice
        elif args.save_best_on == "sliding_window":
            if val_sw_dice is not None:
                metric_for_best = val_sw_dice
        
        print('Meitrc for best model : ', metric_for_best)

        scheduler.step()
        #val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, epoch, args.epochs)

        #if args.val_mode == "patch":
        #    val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, epoch, args.epochs)
        #    
        #elif args.val_mode == "sliding_window":
        #    val_loss, val_dice = validate_one_epoch_sliding_window(model=model, loader=val_loader, criterion=criterion, device=device, epoch=epoch, epochs=args.epochs, 
        #                                                           roi_size=patch_size, sw_batch_size=args.val_sw_batch_size, overlap=args.val_sw_overlap, threshold=0.5,)
        #else:
        #    raise ValueError(f"Unknown val_mode: {args.val_mode}")
        
        print(f"Epoch {epoch:03d}/{args.epochs:03d} - train_loss: {train_loss:.4f}, train_dice: {train_dice:.4f}, val_loss: {val_loss:.4f}, val_dice: {val_dice:.4f}")
        print('print here')
        log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_dice": float(train_dice),
            "val_loss": float(val_loss),
            "val_dice": float(val_dice),
            "val_sw_loss": float(val_sw_loss) if val_sw_loss is not None else None,
            "val_sw_dice": float(val_sw_dice) if val_sw_dice is not None else None,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        
        history.append(log)

        msg = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f}")
        
        if val_sw_dice is not None:
            msg += f" val_sw_loss={val_sw_loss:.4f} val_sw_dice={val_sw_dice:.4f}"
        
        print(msg)

        # save according to the prefered criteria 
        
        print(f"Current best val Dice: {best_val_dice:.4f}")
        print(metric_for_best is not None and metric_for_best > best_val_dice)
        print(metric_for_best > best_val_dice)
        
        
        if metric_for_best is not None and metric_for_best > best_val_dice:
            best_val_dice = metric_for_best
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_val_dice,
                    "val_dice": val_dice,
                    "val_sw_dice": val_sw_dice,
                    "args": vars(args),
                },
                ckpt_dir / "best_model.pt"
            )
            print(f"Saved new best model with metric={metric_for_best:.4f}")
        
        # save last model
        torch.save(
            {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
            "args": vars(args),},
            ckpt_dir / "last_model.pt")

        fp.save_json(history, exp_dir / "history.json")

    print(f"\nTraining finished. Best val Dice = {best_val_dice:.4f}")
    
    ra.analyze_experiment_folder(exp_dir, save_plots=True)
    
    # ------------------------------------------------------------
    # Optional test evaluation using best checkpoint
    # Priority:
    #   1) external test set if provided
    #   2) internal test split if --make_test_split was used
    # ------------------------------------------------------------
    print(f"\nTraining finished. Best val Dice = {best_val_dice:.4f}")

    has_external_test = args.test_images_dir is not None and args.test_masks_dir is not None
    has_internal_test = internal_test_loader is not None and len(internal_test_samples) > 0

    # Test dataset evaluation sliding-window style (using MONAI sliding_window_inference)
    if has_external_test or has_internal_test:
        print("\nLoading best checkpoint for test evaluation...")

        best_ckpt_path = ckpt_dir / "best_model.pt"
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        if has_external_test:
            print("Using external test dataset.")
            test_loader, test_samples = dl.create_test_dataloader(images_dir=args.test_images_dir, masks_dir=args.test_masks_dir, patch_size=patch_size, num_workers=args.num_workers,)
            
        else:
            print("Using internal test split from the main dataset.")
            test_loader = internal_test_loader
            test_samples = internal_test_samples

        print(f"Test cases: {len(test_samples)}")

        rows, summary = evaluate_on_test_set_sliding_window(model=model, loader=test_loader, device=device, output_dir=exp_dir, save_predictions=args.save_test_predictions, 
                                                 threshold=0.5, roi_size=patch_size, sw_batch_size=1, overlap=0.5,)
        
    # Test dataset evaluation NoduleNet-style
    if len(test_samples) > 0:
        print("\nLoading best model for NoduleNet-style ROI test...")

        ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        rows, summary = evaluate_nodulenet_style_roi_test(
            model=model,
            test_samples=test_samples,
            device=device,
            output_dir=exp_dir,
            roi_margin=args.roi_test_margin,
            factor=factor,
            threshold=args.roi_test_threshold,
            save_predictions=args.save_test_predictions,)

        print(f"ROI test results: mean_dice={summary['mean_dice']:.4f}, "
              f"mean_iou={summary['mean_iou']:.4f}")
        print(f"Saved ROI test CSV to: {exp_dir / 'test_metrics_roi.csv'}")                                     

        print(
            f"Test results: mean_dice={summary['mean_dice']:.4f} "
            f"mean_iou={summary['mean_iou']:.4f}"
        )
        print(f"Saved test CSV to: {exp_dir / 'test_metrics.csv'}")

        if args.save_test_predictions:
            print(f"Saved predictions to: {exp_dir / 'predictions'}")
    

if __name__ == "__main__":
    main()