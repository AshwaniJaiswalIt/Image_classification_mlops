# MLOps Assignment – Cats vs Dogs Image Classification

**M1: Model Development & Experiment Tracking**

---

## Project Structure

```
Image_classification_mlops/
├── PetImages/                    ← Raw dataset (DVC-tracked)
│   ├── Cat/                      ← ~12,500 cat images
│   └── Dog/                      ← ~12,500 dog images
├── cats_dogs_dataset/            ← Preprocessed split (cats/ dogs/ subfolders)
│   ├── cats/
│   └── dogs/
├── cats_dogs_code/               ← Source package + unit tests
│   ├── src/
│   │   ├── preprocessing.py      ← Image loading, 224x224 resize, augmentation, split
│   │   └── training.py           ← CNN model build/train/evaluate/save
│   ├── tests/
│   │   ├── test_preprocessing.py ← 9 unit tests (augmentation, loading, splitting)
│   │   └── test_model.py         ← 2 unit tests (build, train, evaluate, save/load)
│   ├── pytest.ini
│   └── .flake8
├── .github/workflows/
│   ├── ci.yml                    ← CI: lint → test → docker build
│   └── ml_pipeline.yml           ← Full ML pipeline: lint → test → train + MLflow
├── train_pipeline.py             ← End-to-end training with MLflow tracking
├── cat_dog_model.h5              ← Trained Keras CNN model
├── PetImages.dvc                 ← DVC tracking file for the dataset
├── requirements.txt
└── README.md
```

---

## M1 Completion Status

| Task | Status | Details |
|------|--------|---------|
| **Git versioning** | ✅ Done | Full git history, `.gitignore` configured |
| **DVC dataset versioning** | ✅ Done | `PetImages.dvc` tracks 24,998 images (848 MB) |
| **Image preprocessing** | ✅ Done | 224×224 RGB resize, 80/10/10 split, 5 augmentations |
| **Baseline CNN model** | ✅ Done | Simple CNN, trained, saved as `cat_dog_model.h5` |
| **MLflow experiment tracking** | ✅ Done | Logs hyperparams, per-epoch metrics, test metrics, confusion matrix, training curves, Keras model artifact |
| **Unit tests** | ✅ Done | 11/11 tests passing |
| **CI/CD pipeline** | ✅ Done | GitHub Actions: lint → test → train |

---

## Quick Start

### 1. Setup Virtual Environment and Install Dependencies

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

The raw dataset is tracked with DVC. Populate `cats_dogs_dataset/` with:
- `cats_dogs_dataset/cats/` → copy images from `PetImages/Cat/`
- `cats_dogs_dataset/dogs/` → copy images from `PetImages/Dog/`

Or use DVC (after configuring a remote):
```powershell
pip install dvc
dvc pull  # pulls cats_dogs_dataset from remote
```

### 3. Train Model with MLflow Tracking

```powershell
python train_pipeline.py
```

**What this does:**
- Loads images from `cats_dogs_dataset/` and resizes to 224×224 RGB
- Splits 80% train / 10% val / 10% test (stratified)
- Applies 5 data augmentation techniques to training set (2× multiplier)
- Builds and trains a simple CNN (Adam optimizer, binary crossentropy)
- Logs to MLflow: all hyperparameters, per-epoch train/val loss & accuracy, test metrics (accuracy, precision, recall, F1, ROC-AUC)
- Saves confusion matrix and training curves as artifacts
- Saves `cat_dog_model.h5` (Keras HDF5 format)

**Output artifacts:**
- `cat_dog_model.h5` — trained model
- `training_config.json` — run configuration and all metrics
- `confusion_matrix.png` — test set confusion matrix
- `training_curves.png` — loss and accuracy curves
- `mlruns/` — MLflow experiment logs

### 4. View MLflow Results

```powershell
mlflow ui
```

Open: [http://localhost:5000](http://localhost:5000)

You will see the experiment `cats_vs_dogs_classification` with:
- All hyperparameters logged
- Per-epoch training and validation loss/accuracy curves
- Final test metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix image
- Training curves image
- Saved Keras model artifact

---

## Unit Tests

### Run All Tests
```powershell
cd cats_dogs_code
python -m pytest tests/ -v
```

### Run with Coverage
```powershell
python -m pytest tests/ -v --cov=src --cov-report=html
```
View coverage report at `cats_dogs_code/htmlcov/index.html`.

### Run Linting
```powershell
flake8 src tests
```

**Test suite (11 tests):**
- `test_model.py` — CNN build/train, evaluate, save/load
- `test_preprocessing.py` — image loading, split, preprocess, augment (shape, range, reproducibility, size, labels, shuffling, metadata)

---

## DVC Dataset Versioning

The dataset is tracked with DVC. `PetImages.dvc` records the MD5 hash, size (848 MB), and file count (24,998 images).

```powershell
# Check DVC status
dvc status

# Add/update dataset tracking
dvc add PetImages

# Configure a remote storage (e.g., S3, GDrive, local)
dvc remote add -d myremote s3://your-bucket/dvc-store
dvc push           # push data to remote
dvc pull           # pull data from remote
```

---

## Data Augmentation

Applied to training set with 2× multiplier:

| Technique | Details |
|-----------|---------|
| Horizontal flip | 50% probability |
| Vertical flip | 50% probability |
| Rotation | ±15°, 50% probability |
| Brightness | 0.8–1.2× factor, 50% probability |
| Contrast | 0.8–1.2× factor, 50% probability |

---

## Model Architecture (Simple CNN)

```
Input: (224, 224, 3)
Conv2D(32, 3×3, relu) → MaxPooling(2×2)
Conv2D(64, 3×3, relu) → MaxPooling(2×2)
Flatten → Dense(64, relu) → Dense(1, sigmoid)

Loss: binary_crossentropy | Optimizer: adam
```

---

## CI/CD Pipeline (GitHub Actions)

**`.github/workflows/ml_pipeline.yml`** (full pipeline):
1. **Lint** — flake8 syntax + style, pylint
2. **Test** — pytest with coverage report (uploaded as artifact)
3. **Train** — runs `train_pipeline.py` with MLflow, uploads `cat_dog_model.h5`, plots, config, and `mlruns/`

**`.github/workflows/ci.yml`** (fast CI):
- lint → test → docker build

---

## Dependencies (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| tensorflow | CNN model (Keras API) |
| pillow, opencv-python | Image processing |
| scikit-learn | Train/val/test splits, metrics |
| mlflow | Experiment tracking |
| numpy, matplotlib | Numerics and plots |
| dvc | Dataset versioning |
| pytest, pytest-cov, flake8 | Testing and linting |
