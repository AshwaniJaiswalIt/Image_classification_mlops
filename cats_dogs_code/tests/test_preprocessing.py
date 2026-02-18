"""Unit tests for image preprocessing functions"""
import os

import numpy as np
from PIL import Image

from src.preprocessing import (
    load_images_from_folder,
    split_dataset,
    preprocess_single_image
)


def create_dummy_image(path, size=(224, 224), color=(0, 0, 0)):
    img = Image.new('RGB', size, color)
    img.save(path)


def test_load_and_split(tmp_path):
    # create minimal folder structure
    cat_dir = tmp_path / "cats"
    dog_dir = tmp_path / "dogs"
    cat_dir.mkdir()
    dog_dir.mkdir()

    for i in range(3):
        create_dummy_image(cat_dir / f"cat{i}.jpg")
        create_dummy_image(dog_dir / f"dog{i}.jpg")

    X, y = load_images_from_folder(str(tmp_path))
    assert X.shape[0] == 6
    assert y.shape[0] == 6
    assert set(y.tolist()) == {0, 1}

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, test_size=0.2, val_size=0.2
    )
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 6
    assert y_train.shape[0] == X_train.shape[0]
    assert y_val.shape[0] == X_val.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


def test_preprocess_single_image(tmp_path):
    img_path = tmp_path / "sample.png"
    create_dummy_image(img_path, size=(100, 100))
    arr = preprocess_single_image(str(img_path), target_size=(50, 50))
    assert arr.shape == (50, 50, 3)
    assert arr.max() <= 1.0
    assert arr.min() >= 0.0