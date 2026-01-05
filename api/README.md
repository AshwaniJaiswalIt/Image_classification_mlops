# Heart Disease Prediction API

FastAPI-based Docker container for serving heart disease prediction model.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- Port 8000 available

### Build Docker Image
```bash
cd api
docker build -t heart-disease-api .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name heart-api heart-disease-api
```

### Test API
```bash
# Run test script
python test_api.py

# Or open browser
http://localhost:8000/docs
```

---

## 📡 API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Returns API health status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_type": "Random Forest"
}
```

### 3. Model Information
```
GET /model/info
```
Returns details about the loaded model.

**Response:**
```json
{
  "model_type": "Random Forest Classifier",
  "n_estimators": 200,
  "features": 13,
  "target_classes": ["No Disease", "Disease"],
  "training_accuracy": "88.5%"
}
```

### 4. Predict (Single)
```
POST /predict
```
Make prediction for a single patient.

**Request Body:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Disease",
  "confidence": 0.8723,
  "probabilities": {
    "no_disease": 0.1277,
    "disease": 0.8723
  },
  "model_used": "Random Forest"
}
```

### 5. Predict (Batch)
```
POST /predict/batch
```
Make predictions for multiple patients.

**Request Body:** Array of patient objects

---

## 🧪 Testing

### Option 1: Python Test Script
```bash
python test_api.py
```

### Option 2: Interactive Documentation
Open browser: `http://localhost:8000/docs`
- Click "Try it out"
- Enter JSON data
- Click "Execute"

### Option 3: curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

---

## 📊 Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| age | Age in years | 0-120 |
| sex | Sex (1=male, 0=female) | 0-1 |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | >0 |
| chol | Serum cholesterol (mg/dl) | >0 |
| fbs | Fasting blood sugar > 120 mg/dl | 0-1 |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate achieved | 0-250 |
| exang | Exercise induced angina | 0-1 |
| oldpeak | ST depression | ≥0 |
| slope | Slope of peak exercise ST | 0-2 |
| ca | Number of major vessels (0-3) | 0-3 |
| thal | Thalassemia | 0-3 |

---

## 🐳 Docker Commands

### Build
```bash
docker build -t heart-disease-api .
```

### Run
```bash
# Foreground
docker run -p 8000:8000 heart-disease-api

# Background
docker run -d -p 8000:8000 --name heart-api heart-disease-api
```

### Manage
```bash
# List running containers
docker ps

# View logs
docker logs heart-api

# Stop container
docker stop heart-api

# Remove container
docker rm heart-api

# Remove image
docker rmi heart-disease-api
```

---

## 📁 Project Structure

```
api/
├── app.py                  # FastAPI application
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── .dockerignore         # Docker build exclusions
├── test_api.py           # API test suite
├── models/               # Model files
│   ├── random_forest_model.pkl
│   └── imputer.pkl
└── README.md            # This file
```

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Use different port
docker run -p 8001:8000 heart-disease-api
```

### Container Not Starting
```bash
# Check logs
docker logs heart-api

# Interactive mode
docker run -it heart-disease-api
```

### Model Not Loading
```bash
# Verify model files exist
docker exec heart-api ls -l models/
```

---

## 📝 Example Usage (Python)

```python
import requests

# Prepare patient data
patient = {
    "age": 63, "sex": 1, "cp": 3,
    "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0,
    "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0,
    "ca": 0, "thal": 1
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json=patient
)

# Print result
result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🎯 Assignment Deliverables

✅ Docker container built and running locally  
✅ `/predict` endpoint accepting JSON input  
✅ Returns prediction and confidence  
✅ Sample input tested successfully  
✅ Interactive API documentation available  

---

## 📄 License

Part of MLOps Assignment - Heart Disease Classification Project
