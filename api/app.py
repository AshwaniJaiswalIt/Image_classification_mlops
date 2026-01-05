"""
Heart Disease Prediction API
FastAPI application for serving ML model predictions
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from typing import Dict
import os

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using Random Forest model",
    version="1.0.0"
)

# Define input data schema
class PatientData(BaseModel):
    age: float = Field(..., description="Age in years", ge=0, le=120)
    sex: float = Field(..., description="Sex (1 = male; 0 = female)", ge=0, le=1)
    cp: float = Field(..., description="Chest pain type (0-3)", ge=0, le=3)
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)", ge=0)
    chol: float = Field(..., description="Serum cholesterol (mg/dl)", ge=0)
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", ge=0, le=1)
    restecg: float = Field(..., description="Resting ECG results (0-2)", ge=0, le=2)
    thalach: float = Field(..., description="Maximum heart rate achieved", ge=0, le=250)
    exang: float = Field(..., description="Exercise induced angina (1 = yes; 0 = no)", ge=0, le=1)
    oldpeak: float = Field(..., description="ST depression induced by exercise", ge=0)
    slope: float = Field(..., description="Slope of peak exercise ST segment (0-2)", ge=0, le=2)
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)", ge=0, le=3)
    thal: float = Field(..., description="Thalassemia (0-3)", ge=0, le=3)
    
    class Config:
        schema_extra = {
            "example": {
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
        }

# Load models at startup
MODEL_DIR = "models"

try:
    print("Loading models...")
    with open(f'{MODEL_DIR}/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open(f'{MODEL_DIR}/imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    rf_model = None
    imputer = None

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Make prediction (POST)",
            "/docs": "Interactive API documentation",
            "/model/info": "Model information"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if API and models are healthy"""
    models_loaded = rf_model is not None and imputer is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "model_type": "Random Forest" if models_loaded else None
    }

# Model info endpoint
@app.get("/model/info")
def model_info():
    """Get information about the loaded model"""
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Classifier",
        "n_estimators": rf_model.n_estimators,
        "features": 13,
        "target_classes": ["No Disease", "Disease"],
        "training_accuracy": "88.5%",
        "description": "Predicts presence of heart disease based on 13 clinical features"
    }

# Prediction endpoint
@app.post("/predict")
def predict(patient: PatientData) -> Dict:
    """
    Make heart disease prediction for a patient
    
    Returns:
        - prediction: 0 (No Disease) or 1 (Disease)
        - prediction_label: Human-readable prediction
        - confidence: Confidence score (0-1)
        - probabilities: Probability for each class
    """
    # Check if models are loaded
    if rf_model is None or imputer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to array
        features = np.array([[
            patient.age, patient.sex, patient.cp, patient.trestbps,
            patient.chol, patient.fbs, patient.restecg, patient.thalach,
            patient.exang, patient.oldpeak, patient.slope, patient.ca,
            patient.thal
        ]])
        
        # Apply imputation (though likely not needed for API input)
        features_imputed = imputer.transform(features)
        
        # Make prediction
        prediction = rf_model.predict(features_imputed)[0]
        probabilities = rf_model.predict_proba(features_imputed)[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[prediction])
        
        # Format response
        response = {
            "prediction": int(prediction),
            "prediction_label": "Disease" if prediction == 1 else "No Disease",
            "confidence": round(confidence, 4),
            "probabilities": {
                "no_disease": round(float(probabilities[0]), 4),
                "disease": round(float(probabilities[1]), 4)
            },
            "model_used": "Random Forest",
            "input_features": {
                "age": patient.age,
                "sex": "Male" if patient.sex == 1 else "Female",
                "chest_pain_type": int(patient.cp),
                "resting_bp": patient.trestbps,
                "cholesterol": patient.chol
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint (bonus)
@app.post("/predict/batch")
def predict_batch(patients: list[PatientData]) -> Dict:
    """
    Make predictions for multiple patients at once
    
    Returns list of predictions
    """
    if rf_model is None or imputer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    for patient in patients:
        try:
            # Reuse single prediction logic
            prediction_result = predict(patient)
            results.append(prediction_result)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {
        "total_patients": len(patients),
        "predictions": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
