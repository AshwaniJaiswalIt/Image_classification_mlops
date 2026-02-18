"""
Image classification training pipeline for Cats vs Dogs dataset.
Saves a simple CNN model and configuration necessary for inference.
"""
import sys
import os
import json

import numpy as np

# add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cats_dogs_code'))

from src.preprocessing import load_images_from_folder, split_dataset
from src.training import build_simple_cnn, train_cnn, evaluate_model, save_model


def main():
    print("=" * 70)
    print("STARTING IMAGE TRAINING PIPELINE")
    print("=" * 70)

    # 1. load and preprocess images
    print("\n[1/5] Loading and preprocessing images...")
    data_dir = './cats_dogs_dataset'  # users must populate this folder via DVC or manual download
    X, y = load_images_from_folder(data_dir)
    print(f"Loaded {len(X)} images")

    # 2. split dataset
    print("\n[2/5] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, test_size=0.1, val_size=0.1)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. build model
    print("\n[3/5] Building model...")
    model = build_simple_cnn(input_shape=X_train.shape[1:])
    model.summary()

    # 4. train model
    print("\n[4/5] Training model (1 epoch for demo)...")
    model = train_cnn(model, X_train, y_train, X_val, y_val, epochs=1)

    # 5. evaluate and save
    print("\n[5/5] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    model_path = 'cat_dog_model.h5'
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    config = {
        'input_shape': list(X_train.shape[1:]),
        'classes': ['cats', 'dogs'],
        'metrics': metrics
    }
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Configuration saved to training_config.json")

    print("\nTraining pipeline completed successfully")


if __name__ == '__main__':
    main()

