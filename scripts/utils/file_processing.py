import csv
import json
from pathlib import Path
from datetime import datetime

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows, path):

    rows = list(rows)
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_extensions(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return Path(filename).stem



def make_experiment_dir(args, root_dir="results"):
    """
    If output_dir is provided, use it.
    Otherwise create:
        results/unet3d_ps<patch>_<timestamp>
    """
    output_dir = args.output_dir

    if output_dir is not None:
        exp_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(root_dir) / f"unet3d_ps{'x'.join(map(str, args.patch_size))}_{timestamp}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir