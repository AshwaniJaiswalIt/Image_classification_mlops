"""Image preprocessing utilities for Cats-vs-Dogs dataset.

This module replaces the previous tabular preprocessing code
used for the heart disease example. Functions here load images,
resize them to the standard 224x224 RGB shape, and prepare
train/validation/test splits for model development.
"""

import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_images_from_folder(
    folder: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all images from a directory structure where subfolders
    correspond to class labels (e.g. 'cats/' and 'dogs/').

    Args:
        folder: Root directory containing class subfolders.
        target_size: Desired image size (width, height).

    Returns:
        Tuple of (images_array, labels_array). Images will have shape
        (n_samples, height, width, 3) and dtype float32 scaled to [0,1].
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    folder_path = Path(folder)
    for label_idx, class_dir in enumerate(sorted(folder_path.iterdir())):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.*"):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(target_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(label_idx)
            except Exception:
                # skip unreadable files
                continue

    if images:
        X = np.stack(images)
        y = np.array(labels, dtype=np.int32)
    else:
        X = np.empty((0, target_size[1], target_size[0], 3), dtype=np.float32)
        y = np.empty((0,), dtype=np.int32)

    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/validation/test sets.

    Args:
        X: Feature array
        y: Label array
        test_size: Proportion held out for test set.
        val_size: Proportion of *remaining* data used for validation.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative,
        random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# helper for unit tests or downstream code

def preprocess_single_image(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Load and preprocess a single image file, returning a scaled array."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

