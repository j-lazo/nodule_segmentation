# ============================================================
# Losses and metrics
# ============================================================
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


@torch.no_grad()
def dice_coefficient(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


@torch.no_grad()
def compute_case_metrics_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """
    probs, targets: [1, 1, D, H, W] or [B, 1, D, H, W]
    Returns lists for each batch element.
    """
    preds = (probs > threshold).float()

    B = preds.shape[0]
    preds = preds.view(B, -1)
    targets = targets.view(B, -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum

    dice = (2.0 * intersection + eps) / (union + eps)
    iou = (intersection + eps) / (pred_sum + target_sum - intersection + eps)

    return {
        "dice": dice.cpu().numpy().tolist(),
        "iou": iou.cpu().numpy().tolist(),
        "pred_voxels": pred_sum.cpu().numpy().tolist(),
        "target_voxels": target_sum.cpu().numpy().tolist(),
    }
    

def dice_numpy(pred, gt, eps=1e-6):
    pred = pred > 0
    gt = gt > 0

    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    return float((2 * inter + eps) / (denom + eps))


def iou_numpy(pred, gt, eps=1e-6):
    pred = pred > 0
    gt = gt > 0

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    return float((inter + eps) / (union + eps))