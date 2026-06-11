import torch
import numpy as np
from pathlib import Path
import random
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from . import image_processing_functions as ip
from . import file_processing as fp


class Luna16PairedRoiDataset(Dataset):
    def __init__(self, samples, patch_size, training = True, positive_crop_prob = 0.8, hu_min = -1000.0, hu_max = 400.0, target_spacing=(1.0, 1.0, 1.0),):
        
        if isinstance(samples, dict):
            self.samples = []
            for case_id, paths in samples.items():
                self.samples.append({
                    "case_id": case_id,
                    "image": paths["path_image"],
                    "mask": paths["path_mask"],
                })
        elif isinstance(samples, list):
            self.samples = samples
        else:
            raise TypeError(f"samples must be dict or list, got {type(samples)}")

        self.patch_size = patch_size
        self.training = training
        self.positive_crop_prob = positive_crop_prob
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_spacing = tuple(target_spacing)
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image, image_spacing = ip.load_mha(item["image"])
        mask, mask_spacing = ip.load_nii_mask(item["mask"])
        original_shape = list(image.shape)
        original_spacing = list(image_spacing)
        
        image = ip.resample_volume_to_spacing(image, current_spacing=image_spacing, target_spacing=self.target_spacing, is_mask=False,)
        mask = ip.resample_volume_to_spacing(mask, current_spacing=mask_spacing, target_spacing=self.target_spacing, is_mask=True)

        image = ip.clip_and_scale_ct(image, self.hu_min, self.hu_max)

        if self.training:
            force_fg = (np.random.rand() < self.positive_crop_prob) and (mask.sum() > 0)

            image, mask = ip.center_or_random_crop(image=image, mask=mask, crop_size=self.patch_size, force_foreground=force_fg,)
            image, mask = ip.random_flip_3d(image, mask)
            image = ip.random_intensity_shift_scale(image)

        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.float32)).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "case_id": item["case_id"],
            "image_path": item["image"],
            "mask_path": item["mask"],
            "original_shape": original_shape,
            "original_spacing": original_spacing,
            "target_spacing": list(self.target_spacing),
        }


class LungNodule3DDataset(Dataset):
    def __init__(
        self,
        samples,
        patch_size=(128, 128, 128),
        training=True,
        use_patch_sampling=True,
        positive_crop_prob=0.7,
        hu_min=-1000.0,
        hu_max=400.0,
        normalize_mode="clip",
        full_volume_inference=False,):
        
        self.samples = samples
        self.patch_size = patch_size
        self.training = training
        self.use_patch_sampling = use_patch_sampling
        self.positive_crop_prob = positive_crop_prob
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.normalize_mode = normalize_mode
        self.full_volume_inference = full_volume_inference

    def __len__(self):
        return len(self.samples)

    def _normalize(self, image):
        if self.normalize_mode == "clip":
            return ip.clip_and_scale_ct(image, self.hu_min, self.hu_max)
        elif self.normalize_mode == "zscore":
            return ip.zscore_nonzero(image)
        raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image, _ = ip.load_mha(item["image"])
        mask, _ = ip.load_nii_mask(item["mask"])

        original_shape = list(image.shape)
        image = self._normalize(image)

        if self.training:
            if self.use_patch_sampling:
                force_fg = (random.random() < self.positive_crop_prob) and (mask.sum() > 0)
                image, mask = ip.center_or_random_crop(image, mask, self.patch_size, force_foreground=force_fg)
                
            else:
                image = ip.resize_3d_numpy(image, self.patch_size, is_mask=False)
                mask = ip.resize_3d_numpy(mask, self.patch_size, is_mask=True)

            # augmentation
            image, mask = ip.random_flip_3d(image, mask)
            image = ip.random_intensity_shift_scale(image)

        else:
            # For inference/eval:
            # - if full_volume_inference=True, keep original full volume
            # - otherwise use old resize behavior
            if not self.full_volume_inference:
                image = ip.resize_3d_numpy(image, self.patch_size, is_mask=False)
                mask = ip.resize_3d_numpy(mask, self.patch_size, is_mask=True)

        image = torch.from_numpy(image).float().unsqueeze(0)  # [1, D, H, W]
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # [1, D, H, W]

        return {
            "image": image,
            "mask": mask,
            "case_id": item["case_id"],
            "image_path": item["image"],
            "mask_path": item["mask"],
            "original_shape": original_shape,
        }

"""def create_dataloaders(images_dir, masks_dir, patch_size, batch_size, num_workers, val_ratio, test_ratio=0.0, seed=42, val_mode="patch",):
    
    all_samples = build_file_pairs(images_dir, masks_dir)
    
    train_samples, val_samples, test_samples = split_dataset(all_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    train_ds = LungNodule3DDataset(train_samples, patch_size=patch_size, training=True, use_patch_sampling=True, positive_crop_prob=0.9, normalize_mode="clip",)

    if val_mode == "patch":
        val_ds = LungNodule3DDataset(val_samples, patch_size=patch_size, training=True, use_patch_sampling=True, positive_crop_prob=1.0, normalize_mode="clip",)
        
    elif val_mode == "sliding_window":
        val_ds = LungNodule3DDataset(val_samples, patch_size=patch_size, training=False, use_patch_sampling=False, normalize_mode="clip", full_volume_inference=True,)
    else:
        raise ValueError(f"Unknown val_mode: {val_mode}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)

    test_loader = None
    
    if len(test_samples) > 0:
        test_ds = LungNodule3DDataset(test_samples, patch_size=patch_size, training=False, use_patch_sampling=False, normalize_mode="clip", full_volume_inference=True,)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)

    return train_loader, val_loader, test_loader, train_samples, val_samples, test_samples"""


def create_dataloaders_luna25(images_dir, masks_dir, patch_size, batch_size, num_workers, val_ratio, test_ratio=0.0, seed=42, 
                              train_positive_crop_prob=1, val_positive_crop_prob=0.7):
    
    all_samples = build_file_pairs(images_dir, masks_dir)
    train_samples, val_samples, test_samples = split_dataset(all_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    train_ds = LungNodule3DDataset(train_samples, patch_size=patch_size, training=True, use_patch_sampling=True, positive_crop_prob=train_positive_crop_prob, normalize_mode="clip",)
    val_ds_patch = LungNodule3DDataset(val_samples, patch_size=patch_size, training=True, use_patch_sampling=True, positive_crop_prob=val_positive_crop_prob, normalize_mode="clip",)
    val_ds_sw = LungNodule3DDataset(val_samples, patch_size=patch_size, training=False, use_patch_sampling=False, normalize_mode="clip", full_volume_inference=True,)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader_patch = DataLoader(val_ds_patch, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)

    val_loader_sw = DataLoader(val_ds_sw, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)

  
    if len(test_samples) > 0:
        test_ds = LungNodule3DDataset(test_samples, patch_size=patch_size, training=False, use_patch_sampling=False, normalize_mode="clip", full_volume_inference=True,)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)

    return train_loader, val_loader_patch, val_loader_sw, test_loader, train_samples, val_samples, test_samples


def create_dataloaders_luna16(path_volumes, path_masks, path_ids_lin_file, patch_size, batch_size, num_workers, train_ratio=0.7,
                              val_ratio=0.2, test_ratio=0.1, seed=42, train_positive_crop_prob=0.8, val_positive_crop_prob=0.5,):
    
    
    dict_pairs = create_image_maks_pairs(path_volumes, path_masks, path_ids_lin_file)
    train_samples, val_samples, test_samples = split_dict_pairs(dict_pairs, train_size=train_ratio, val_size=val_ratio, test_size=test_ratio, seed=seed)

    train_ds = Luna16PairedRoiDataset(train_samples, patch_size=patch_size, training=True, positive_crop_prob=train_positive_crop_prob,)
    val_ds = Luna16PairedRoiDataset(val_samples,patch_size=patch_size, training=True, positive_crop_prob=val_positive_crop_prob,)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(),)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),)

    return train_loader, val_loader, train_samples, val_samples, test_samples


def create_test_dataloader(images_dir, masks_dir, patch_size, num_workers):
    
    test_samples = build_file_pairs(images_dir, masks_dir)
    test_ds = LungNodule3DDataset(test_samples, patch_size=patch_size, training=False, use_patch_sampling=False, normalize_mode="clip", full_volume_inference=True,)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,)
    
    return test_loader, test_samples



def create_image_maks_pairs(path_volumes, path_masks, path_ids_lin_file):
    
    df_ids_link = pd.read_csv(path_ids_lin_file)
    list_volumes = os.listdir(path_volumes)
    mhd_files = [f for f in list_volumes if f.endswith('.mhd')]
    only_name_files = [f.replace('.mhd', '') for f in mhd_files]
    
    list_files_masks = os.listdir(path_masks)
    list_maks_nodules = [f for f in list_files_masks if 'mask' in f and 'contour' in f and 'circle' not in f and 'nodule' in f]

    list_number_masks = [int(s.split('_')[0]) for s in list_maks_nodules]
    print(len(list_maks_nodules), 'Masks in ', path_masks)    
    print(len(only_name_files), 'CT Volumes in ', path_volumes)
    
    output_dict = {}
    
    for mhd_file in tqdm(only_name_files):
        sub_df_links = df_ids_link[df_ids_link["SeriesID"] == mhd_file]
        if len(sub_df_links) == 0:
            print("No mask link found for:", mhd_file)
                    
        mask_id_num = sub_df_links["CID"].tolist()[0]
        if mask_id_num not in list_number_masks:
            print("Mask id not found:", mask_id_num, mhd_file)
            continue 
        
        idx = list_number_masks.index(mask_id_num)
        path_mask = os.path.join(path_masks, list_maks_nodules[idx])
        path_image = os.path.join(path_volumes, mhd_file + ".mhd")
        output_dict[mhd_file]  = {'path_image': path_image, 
                                  'path_mask': path_mask}
        
    return output_dict 


def split_dict_pairs(dict_pairs, train_size=0.70, val_size=0.20, test_size=0.1, seed=42,):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    keys = sorted(list(dict_pairs.keys()))

    train_keys, temp_keys = train_test_split(
        keys,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
    )

    relative_val_size = val_size / (val_size + test_size)

    val_keys, test_keys = train_test_split(
        temp_keys,
        train_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )

    train_pairs = {k: dict_pairs[k] for k in train_keys}
    val_pairs   = {k: dict_pairs[k] for k in val_keys}
    test_pairs  = {k: dict_pairs[k] for k in test_keys}

    return train_pairs, val_pairs, test_pairs


def build_file_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = sorted(images_dir.glob("*.mha"))
    mask_files = sorted(masks_dir.glob("*.nii.gz"))

    image_map = {fp.strip_extensions(p.name): str(p) for p in image_files}
    mask_map = {fp.strip_extensions(p.name): str(p) for p in mask_files}

    common_ids = sorted(set(image_map.keys()) & set(mask_map.keys()))
    if len(common_ids) == 0:
        raise RuntimeError(
            f"No matching image/mask pairs found.\n"
            f"images_dir={images_dir}\n"
            f"masks_dir={masks_dir}"
        )

    samples = []
    for cid in common_ids:
        samples.append({
            "case_id": cid,
            "image": image_map[cid],
            "mask": mask_map[cid],
        })

    return samples


def split_dataset(samples, val_ratio=0.2, test_ratio= 0.0, seed: int = 42):
    """
    Split dataset into train/val/test.

    If test_ratio == 0, test set will be empty.
    Ratios are fractions of the full dataset.
    """
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be >= 0.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    rng = random.Random(seed)
    samples = samples.copy()
    rng.shuffle(samples)

    n_total = len(samples)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    # keep at least 1 val/test sample if ratios > 0 and dataset is large enough
    if test_ratio > 0 and n_test == 0 and n_total >= 3:
        n_test = 1
    if val_ratio > 0 and n_val == 0 and n_total >= 3:
        n_val = 1

    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]

    if len(train_samples) == 0:
        raise RuntimeError("Training set is empty after split. Reduce val_ratio/test_ratio.")

    return train_samples, val_samples, test_samples

