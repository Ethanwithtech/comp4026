"""
Dataset utilities for Pins Face Recognition dataset.
Handles downloading, loading, splitting, and augmentation.
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_dataset() -> str:
    """Download the Pins Face Recognition dataset from Kaggle."""
    import kagglehub
    path = kagglehub.dataset_download(config.DATASET_NAME)
    print(f"Dataset downloaded to: {path}")
    return path


def discover_dataset(data_path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Discover the dataset structure.
    Returns:
        identity_images: dict mapping identity_name -> list of image paths
        class_names: sorted list of identity names
    """
    identity_images = defaultdict(list)

    # The Pins dataset structure: root/pins_<person_name>/image_files
    # Walk through the dataset directory
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Get the identity from the parent folder name
                identity = os.path.basename(root)
                if identity:
                    full_path = os.path.join(root, f)
                    identity_images[identity].append(full_path)

    # Filter identities with minimum number of images
    filtered = {}
    for name, paths in identity_images.items():
        if len(paths) >= config.MIN_IMAGES_PER_IDENTITY:
            filtered[name] = sorted(paths)

    # Optionally limit number of identities
    if config.NUM_IDENTITIES is not None:
        keys = sorted(filtered.keys())[:config.NUM_IDENTITIES]
        filtered = {k: filtered[k] for k in keys}

    class_names = sorted(filtered.keys())
    print(f"Found {len(class_names)} identities with >= {config.MIN_IMAGES_PER_IDENTITY} images each")
    total_images = sum(len(v) for v in filtered.values())
    print(f"Total images: {total_images}")

    return filtered, class_names


def split_dataset(
    identity_images: Dict[str, List[str]],
    class_names: List[str],
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test sets.
    Returns lists of (image_path, label_index) tuples.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_data = []
    val_data = []
    test_data = []

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for name in class_names:
        paths = identity_images[name][:]
        random.shuffle(paths)
        n = len(paths)
        n_train = max(1, int(n * config.TRAIN_RATIO))
        n_val = max(1, int(n * config.VAL_RATIO))

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        label = name_to_idx[name]
        train_data.extend([(p, label) for p in train_paths])
        val_data.extend([(p, label) for p in val_paths])
        test_data.extend([(p, label) for p in test_paths])

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


class FaceDataset(Dataset):
    """PyTorch Dataset for face recognition."""

    def __init__(
        self,
        data: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        is_training: bool = False,
    ):
        self.data = data
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_training: bool = False) -> transforms.Compose:
    """Get image transforms for training or evaluation."""
    if is_training and config.USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def get_dataloaders(
    data_path: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train/val/test dataloaders from the dataset.
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    identity_images, class_names = discover_dataset(data_path)
    train_data, val_data, test_data = split_dataset(identity_images, class_names)

    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)

    train_dataset = FaceDataset(train_data, transform=train_transform, is_training=True)
    val_dataset = FaceDataset(val_data, transform=eval_transform)
    test_dataset = FaceDataset(test_data, transform=eval_transform)

    # Weighted sampler for class imbalance
    labels = [d[1] for d in train_data]
    class_counts = np.bincount(labels, minlength=len(class_names))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


class AnonymisedFaceDataset(Dataset):
    """
    Dataset for evaluating anonymised images.
    Used by Student B's anonymisation model output.
    """

    def __init__(
        self,
        image_dir: str,
        label_file: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            image_dir: Directory containing anonymised images
            label_file: Optional CSV file with columns [filename, original_identity]
            transform: Image transforms
        """
        self.image_dir = image_dir
        self.transform = transform or get_transforms(is_training=False)
        self.samples = []

        if label_file and os.path.exists(label_file):
            import pandas as pd
            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                img_path = os.path.join(image_dir, row['filename'])
                if os.path.exists(img_path):
                    self.samples.append((img_path, row['original_identity']))
        else:
            # Load all images without labels
            for f in sorted(os.listdir(image_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(image_dir, f), -1))

        print(f"Loaded {len(self.samples)} anonymised images from {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path
