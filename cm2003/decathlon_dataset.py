import hashlib
import os
import sys
import tarfile
import urllib.request

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    Resized,
    EnsureTyped,
)
from torch.utils.data import DataLoader
import torch

def get_transform():
    """
    Returns the composition of transforms to be applied to the dataset.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # labels given explicitly so that MONAI 1.5 and 1.6 behave the same
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityd(keys="image"),
            Resized(keys=["image", "label"], spatial_size=(32, 64, 32), mode=("trilinear", "nearest")),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

def custom_collate(batch):
    """
    Custom collate function that turns a batch of 3D volumes into a batch of 2D slices.

    Each volume has shape (1, 32, 64, 32) (channel, x, y, z). We cut it into its 32 slices along the
    last axis, so a batch of B volumes becomes a batch of B * 32 images of shape (1, 32, 64). The
    MetaTensors are also converted to regular tensors.
    """
    images = []
    labels = []
    for item in batch:
        # Convert to regular tensor and permute dimensions
        images.append(item['image'].as_tensor().permute(3, 0, 1, 2))
        labels.append(item['label'].as_tensor().permute(3, 0, 1, 2))

    return {
        'image': torch.cat(images, dim=0),
        'label': torch.cat(labels, dim=0)
    }

def download_decathlon(root_dir, task):
    """
    Downloads and extracts a Medical Segmentation Decathlon task into root_dir, unless it is
    already there.

    MONAI can download the data itself (DecathlonDataset(download=True)), but that needs the
    optional `requests` package, so we use Python's standard library instead. The download is
    checked against the MD5 checksum that MONAI provides.
    """
    task_dir = os.path.join(root_dir, task)
    if os.path.isdir(os.path.join(task_dir, "imagesTr")):
        return
    os.makedirs(root_dir, exist_ok=True)
    url = monai.apps.DecathlonDataset.resource[task]
    expected_md5 = monai.apps.DecathlonDataset.md5[task]
    tar_path = os.path.join(root_dir, f"{task}.tar")

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, tar_path + ".part")
    with open(tar_path + ".part", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    if md5 != expected_md5:
        os.remove(tar_path + ".part")
        raise RuntimeError(f"The download of {url} is corrupted (MD5 {md5}), please run the cell again.")
    os.replace(tar_path + ".part", tar_path)

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path) as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(root_dir, filter="data")
        else:
            tar.extractall(root_dir)
    os.remove(tar_path)

def get_decathlon_dataloader(root_dir, task, section, batch_size, num_workers, shuffle=False):
    """
    Creates and returns a DataLoader for the Decathlon dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        task (str): Task name (e.g., "Task04_Hippocampus").
        section (str): Dataset section ("training" or "validation"). The validation set is a fixed
            20% of the labelled Medical Segmentation Decathlon training data.
        batch_size (int): Number of 3D volumes per batch (each gives 32 2D slices).
        num_workers (int): Number of worker processes for data loading.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader for the specified Decathlon dataset.
    """
    transform = get_transform()
    download_decathlon(root_dir, task)

    dataset = monai.apps.DecathlonDataset(
        root_dir=root_dir,
        task=task,
        section=section,
        transform=transform,
        download=False,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate
    )



# ----------------------------------------------------------------------------------------------
# 3D pipeline (used in the second part of MONAI.ipynb)
# ----------------------------------------------------------------------------------------------

PATCH_SIZE = (32, 32, 32)


def get_3d_transforms(train, augmentations=None):
    """
    Returns the MONAI transforms for training a 3D model on the hippocampus volumes.

    Unlike get_transform(), the volumes are not resized: they keep their native 1 mm resolution
    (between 24 and 59 voxels along each axis). During training, the model sees random 32x32x32
    patches; for validation it gets whole volumes, which we segment with sliding window inference.

    Args:
        train (bool): True for the training transforms (random patches and augmentation), False
            for validation (whole volumes, no randomness).
        augmentations (list): Optional extra MONAI dictionary transforms for data augmentation,
            applied to the training patches (e.g. RandFlipd, RandAffined, RandScaleIntensityd).
    """
    from monai.transforms import NormalizeIntensityd, RandCropByPosNegLabeld, SpatialPadd

    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # Z-score normalization of the MRI intensities (inside the brain/non-zero region)
        NormalizeIntensityd(keys="image", nonzero=True),
    ]
    if train:
        transforms += [
            # Make sure every volume is at least as large as a patch
            SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
            # Sample 4 patches per volume, half of them centred on the hippocampus (pos) and half on
            # the background (neg), so the model sees enough of the small foreground structure
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE,
                pos=1, neg=1, num_samples=4,
            ),
        ]
        transforms += list(augmentations or [])
    transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
    return Compose(transforms)


def get_decathlon_3d_dataloader(root_dir, task, section, batch_size, num_workers=0, augmentations=None):
    """
    Creates a DataLoader of 3D volumes (validation) or 3D patches (training).

    Args:
        root_dir (str): Root directory of the dataset.
        task (str): Task name (e.g., "Task04_Hippocampus").
        section (str): "training" or "validation".
        batch_size (int): Number of volumes per batch. For training, each volume gives 4 patches, so
            a batch contains 4 * batch_size patches. For validation, use batch_size=1, since the
            volumes have different sizes.
        num_workers (int): Number of worker processes for data loading.
        augmentations (list): Optional data augmentation transforms (training only).
    """
    download_decathlon(root_dir, task)
    train = section == "training"
    dataset = monai.apps.DecathlonDataset(
        root_dir=root_dir,
        task=task,
        section=section,
        transform=get_3d_transforms(train, augmentations),
        download=False,
    )
    # monai.data.DataLoader collates the lists of patches from RandCropByPosNegLabeld into one batch
    return monai.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
