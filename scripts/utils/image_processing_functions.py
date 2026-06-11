import numpy as np
import torch
import random
import SimpleITK as sitk
import torch.nn.functional as F


def bbox_from_binary_mask(mask):
    """
    Return bbox as (z0, y0, x0, z1, y1, x1), where z1/y1/x1 are exclusive.
    """
    coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        return None

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    return int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)


def expand_box(box, margin, shape,):
    """
    Expand bbox by margin and clip to volume shape.
    """
    if isinstance(margin, int):
        mz = my = mx = margin
    else:
        mz, my, mx = margin

    z0, y0, x0, z1, y1, x1 = box
    D, H, W = shape

    z0 = max(0, z0 - mz)
    y0 = max(0, y0 - my)
    x0 = max(0, x0 - mx)

    z1 = min(D, z1 + mz)
    y1 = min(H, y1 + my)
    x1 = min(W, x1 + mx)

    return int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)


def make_box_factor_compatible(box, shape, factor=16,):
    """
    Expand box so each side length is divisible by factor.
    Similar motivation to NoduleNet ext2factor.
    """
    z0, y0, x0, z1, y1, x1 = box
    D, H, W = shape

    dz = z1 - z0
    dy = y1 - y0
    dx = x1 - x0

    new_dz = int(np.ceil(dz / factor) * factor)
    new_dy = int(np.ceil(dy / factor) * factor)
    new_dx = int(np.ceil(dx / factor) * factor)

    add_z = new_dz - dz
    add_y = new_dy - dy
    add_x = new_dx - dx

    z0 -= add_z // 2
    z1 += add_z - add_z // 2

    y0 -= add_y // 2
    y1 += add_y - add_y // 2

    x0 -= add_x // 2
    x1 += add_x - add_x // 2

    # Shift back inside image if needed
    if z0 < 0:
        z1 -= z0
        z0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x0 < 0:
        x1 -= x0
        x0 = 0

    if z1 > D:
        z0 -= z1 - D
        z1 = D
    if y1 > H:
        y0 -= y1 - H
        y1 = H
    if x1 > W:
        x0 -= x1 - W
        x1 = W

    z0 = max(0, z0)
    y0 = max(0, y0)
    x0 = max(0, x0)

    return int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)


def get_oracle_candidate_boxes_from_mask(mask, margin=32, factor=16,):
    """
    Simple oracle ROI extraction.
    Uses one bbox around the whole foreground mask.
    """
    box = bbox_from_binary_mask(mask)
    if box is None:
        return []

    box = expand_box(box, margin=margin, shape=mask.shape)
    box = make_box_factor_compatible(box, shape=mask.shape, factor=factor)

    return [box]


def save_prediction_like_reference(pred_mask, reference_image_path, output_path):
    ref = sitk.ReadImage(str(reference_image_path))

    out = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())

    sitk.WriteImage(out, str(output_path))


def clip_and_scale_ct(image, hu_min=-1000.0, hu_max=400.0):
    """
    Typical CT windowing then scale to [0, 1].
    """
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return image.astype(np.float32)


def resize_3d_numpy(volume,  out_size, is_mask=False,):
    """
    Resize [D, H, W] using torch interpolate.
    """
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]
    mode = "nearest" if is_mask else "trilinear"
    resized = F.interpolate(tensor, size=out_size, mode=mode, align_corners=False if mode == "trilinear" else None)
    arr = resized.squeeze(0).squeeze(0).cpu().numpy()
    if is_mask:
        arr = (arr > 0.5).astype(np.uint8)
    return arr


def load_mha(path):
    """
    Returns:
        image: np.ndarray of shape [D, H, W], float32
        spacing: tuple (z, y, x)
    """
    itk_img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)  # [D, H, W]
    spacing_xyz = itk_img.GetSpacing()  # (x, y, z)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return arr, spacing_zyx


def load_nii_mask(path):
    """
    Returns:
        mask: np.ndarray of shape [D, H, W], uint8
        spacing: tuple (z, y, x)
    """
    itk_img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(itk_img)  # [D, H, W]
    arr = (arr > 0).astype(np.uint8)
    spacing_xyz = itk_img.GetSpacing()  # (x, y, z)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return arr, spacing_zyx


def resample_volume_to_spacing(volume, current_spacing, target_spacing, is_mask=False,):
    """
    Resample a [D,H,W] volume from current_spacing to target_spacing.

    current_spacing and target_spacing must be in (z,y,x) order.

    Uses:
      - trilinear interpolation for CT images
      - nearest-neighbor interpolation for masks
    """
    current_spacing = np.asarray(current_spacing, dtype=np.float32)
    target_spacing = np.asarray(target_spacing, dtype=np.float32)

    old_shape = np.asarray(volume.shape, dtype=np.int32)

    scale = current_spacing / target_spacing
    new_shape = np.round(old_shape * scale).astype(np.int32)
    new_shape = np.maximum(new_shape, 1)

    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)

    if is_mask:
        out = F.interpolate(
            tensor,
            size=tuple(new_shape.tolist()),
            mode="nearest",
        )
    else:
        out = F.interpolate(
            tensor,
            size=tuple(new_shape.tolist()),
            mode="trilinear",
            align_corners=False,
        )

    out = out.squeeze(0).squeeze(0).cpu().numpy()

    if is_mask:
        out = (out > 0.5).astype(np.uint8)
    else:
        out = out.astype(np.float32)

    return out


def random_flip_3d(image: np.ndarray, mask: np.ndarray):
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
    return image, mask


def random_intensity_shift_scale(image: np.ndarray, shift_range=0.1, scale_range=0.1):
    scale = 1.0 + random.uniform(-scale_range, scale_range)
    shift = random.uniform(-shift_range, shift_range)
    return image * scale + shift


def center_or_random_crop(image, mask, crop_size, force_foreground=False):
    """
    Crop [D, H, W] to crop_size.
    If force_foreground=True and mask has positives, crop around a positive voxel.
    """
    D, H, W = image.shape
    cd, ch, cw = crop_size

    if D < cd or H < ch or W < cw:
        pad_d = max(0, cd - D)
        pad_h = max(0, ch - H)
        pad_w = max(0, cw - W)

        pad_before = (pad_d // 2, pad_h // 2, pad_w // 2)
        pad_after = (pad_d - pad_before[0], pad_h - pad_before[1], pad_w - pad_before[2])

        image = np.pad(image, ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])), mode="constant")
        mask = np.pad(mask, ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])), mode="constant")
        D, H, W = image.shape

    if force_foreground and mask.sum() > 0:
        fg = np.argwhere(mask > 0)
        cz, cy, cx = fg[random.randint(0, len(fg) - 1)]
        z1 = np.clip(cz - cd // 2, 0, D - cd)
        y1 = np.clip(cy - ch // 2, 0, H - ch)
        x1 = np.clip(cx - cw // 2, 0, W - cw)
    else:
        z1 = random.randint(0, D - cd) if D > cd else 0
        y1 = random.randint(0, H - ch) if H > ch else 0
        x1 = random.randint(0, W - cw) if W > cw else 0

    image = image[z1:z1 + cd, y1:y1 + ch, x1:x1 + cw]
    mask = mask[z1:z1 + cd, y1:y1 + ch, x1:x1 + cw]
    return image, mask


def save_prediction_nifti(pred_mask: np.ndarray, reference_image_path: str, out_path: str):
    """
    Save prediction [D, H, W] as NIfTI using spatial metadata from reference image.
    pred_mask should be binary or integer array.
    """
    ref_img = sitk.ReadImage(reference_image_path)

    pred_itk = sitk.GetImageFromArray(pred_mask.astype(np.uint8))  # array is [D,H,W]
    pred_itk.SetSpacing(ref_img.GetSpacing())
    pred_itk.SetOrigin(ref_img.GetOrigin())
    pred_itk.SetDirection(ref_img.GetDirection())

    sitk.WriteImage(pred_itk, out_path)