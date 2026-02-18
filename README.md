# MLOps Assignment – Cats vs Dogs Image Classification

## Project Structure
```
Image_classification_mlops/
├── COMMANDS.md                    ← Quick reference
├── train_pipeline.py              ← Script to train CNN and save artifacts
├── requirements.txt               ← All dependencies (including TF, Pillow, DVC)
├── cats_dogs_dataset/             ← DVC‑tracked images (not in Git)
├── cats_dogs_code/            ← renamed folder containing image utilities & tests
│   └── COMMANDS.md                ← Testing/CI instructions
├── api/                           ← FastAPI inference service
│   └── COMMANDS.md                ← Docker commands
└── kubernetes/                    ← Kubernetes manifests
    └── COMMANDS.md                ← Deployment instructions
```
## For final report and Screenshots please refer the 'screenshorts' folder

---

## Quick Start (First Time Setup - REQUIRED!)

### 1. Create Virtual Environment & Install ALL Packages
```powershell
# Step 1: Create virtual environment
python -m venv venv

# Step 2: Activate it
.\venv\Scripts\Activate.ps1

# Step 3: Install ALL dependencies (includes pytest, flake8, jupyter, etc.)
pip install -r requirements.txt
```

**IMPORTANT:** You MUST run `pip install -r requirements.txt` before running any commands!

This installs:
- pytest (for testing)
- flake8 (for linting)
- jupyter (for notebook)
- All ML libraries

---

## Task 1-4: Model Development & Experiment Tracking

### Data Versioning
The image dataset is large and is tracked with DVC. After cloning the repo:
```powershell
cd Image_classification_mlops
pip install dvc
# pull data (requires internet)
dvc pull cats_dogs_dataset
```

### Run Notebook (if provided)
*This project does not rely on a notebook; the training script (`train_pipeline.py`) performs all steps.*

### Train Model
```powershell
python train_pipeline.py
```
This will load images from `cats_dogs_dataset/`, build a simple CNN, train for 1 epoch, and
write `cat_dog_model.h5` and `training_config.json`.

### View MLflow Results (if enabled)
```powershell
mlflow ui
```
Open: http://localhost:5000

**Output Files:**
- `cat_dog_model.h5` – Trained Keras model
- `training_config.json` – Run configuration and metrics

---

## Task 2-5: CI/CD & Testing

### Navigate to Code Folder
```powershell
cd cats_dogs_code
```

### Run Unit Tests
```powershell
pytest -v
```
(should exercise the image preprocessing and training utilities)

### Run Tests with Coverage
```powershell
pytest -v --cov=src --cov-report=html
```
See `htmlcov/index.html` for coverage reports.

### Run Linting
```powershell
flake8 src tests
```

### Go Back to Root
```powershell
cd ..
```

### CI Pipeline (see section below)

The CI configuration (GitHub Actions) installs dependencies, runs lint/tests, and builds the
Docker image. The workflow file is located at `.github/workflows/ci.yml`.

---

## Task 6: Docker API

### Navigate to API Folder
```powershell
cd api
```

### Build Docker Image
```powershell
# replace <your-tag> with appropriate name, e.g. cats-dogs-api
docker build -t cats-dogs-api .
```

### Run Container
```powershell
docker run -d -p 8000:8000 --name cats-api cats-dogs-api
```

### Test API
```powershell
# health
curl http://localhost:8000/health
# prediction (upload a file)
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

### Run Tests
```powershell
python test_api.py
```

### Stop Container
```powershell
docker stop cats-api
docker rm cats-api
```
http://localhost:8000/docs

# Or run test script
python test_api.py

# Or check health
curl http://localhost:8000/health
```

### View Metrics & Logs
```powershell
# View API metrics
http://localhost:8000/metrics

# View request logs
http://localhost:8000/logs
```

### Stop Container
```powershell
docker stop heart-api
docker rm heart-api
```

---

## Task 7: Kubernetes Deployment (Minikube)

### Start Minikube
```powershell
minikube start
```

### Build Image in Minikube
```powershell
# Point Docker to Minikube's Docker
minikube docker-env | Invoke-Expression

# Build image
cd api
docker build -t heart-disease-api:v1 .
cd ..
```

### Deploy to Kubernetes
```powershell
# Apply deployment and service
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

### Update After Code Changes
```powershell
# Rebuild image in Minikube
minikube docker-env | Invoke-Expression
cd api
docker build -t heart-disease-api:v1 .
cd ..

# Restart deployment
kubectl rollout restart deployment heart-disease-deployment
```

### Verify Deployment
```powershell
# Check pods
kubectl get pods

# Check deployment
kubectl get deployments

# Check service
kubectl get services
```

### Access the API
```powershell
# Open service in browser
minikube service heart-disease-service

# Or get URL
minikube service heart-disease-service --url
```


### Screenshots to Take
```powershell
kubectl get pods -o wide
kubectl get deployment heart-disease-deployment
kubectl get service heart-disease-service
# Also: Browser screenshots of /docs and /health endpoints
```

### Cleanup
```powershell
kubectl delete -f kubernetes/
minikube stop
```

---

## Assignment Tasks Checklist

- [ ] Task 1: Data Acquisition & EDA (5 marks)
  - Run: `jupyter notebook assignment1.ipynb` (Cells 1-28)

- [ ] Task 2: Models (8 marks)
  - Run: Notebook cells 29-42

- [ ] Task 3: MLflow (5 marks)
  - Run: Notebook cells 43-46
  - View: `mlflow ui`

- [ ] Task 4: Packaging (7 marks)
  - Run: Notebook cells 47-51
  - Files: `*.pkl`, `preprocessing_config.json`

- [ ] Task 5: CI/CD (8 marks)
  - Run: `cd cats_dogs_code && pytest -v`
  - GitHub Actions: Auto-runs on push

- [ ] Task 6: Docker API
  - Run: `cd api && docker build -t heart-disease-api .`
  - Test: `docker run -p 8000:8000 heart-disease-api`

- [ ] Task 7: Kubernetes Deployment (7 marks)
  - Run: `minikube start`
  - Deploy: `kubectl apply -f kubernetes/`
  - Access: `minikube service heart-disease-service`

- [ ] Monitoring: Logging & Metrics
  - Metrics: `http://localhost:8000/metrics`
  - Logs: `http://localhost:8000/logs`
  - File: `api/api_logs.log`

---

## Detailed Commands

See folder-specific COMMANDS.md:
- `cats_dogs_code/COMMANDS.md` - Testing details
- `api/COMMANDS.md` - Docker details
- `kubernetes/COMMANDS.md` - Minikube deployment details

---

## Dependencies (in requirements.txt)

Core ML:
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- mlflow

Testing & Quality:
- pytest, pytest-cov
- flake8, pylint

Notebook:
- jupyter
