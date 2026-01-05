"""
Test script for Heart Disease Prediction API
Run this after starting the Docker container
"""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed!")

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("TEST 2: Model Information")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Model info retrieved successfully!")

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*70)
    print("TEST 3: Single Prediction")
    print("="*70)
    
    # Sample patient data (high risk)
    patient_data = {
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
    
    print("Input:")
    print(json.dumps(patient_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    print("✅ Prediction successful!")

def test_healthy_patient():
    """Test prediction for healthy patient"""
    print("\n" + "="*70)
    print("TEST 4: Healthy Patient Prediction")
    print("="*70)
    
    # Sample patient data (low risk)
    patient_data = {
        "age": 35,
        "sex": 0,
        "cp": 0,
        "trestbps": 120,
        "chol": 180,
        "fbs": 0,
        "restecg": 0,
        "thalach": 170,
        "exang": 0,
        "oldpeak": 0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    print("Input (Healthy Profile):")
    print(json.dumps(patient_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ Healthy patient prediction successful!")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*70)
    print("TEST 5: Batch Prediction")
    print("="*70)
    
    patients = [
        {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        },
        {
            "age": 35, "sex": 0, "cp": 0, "trestbps": 120, "chol": 180,
            "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
            "oldpeak": 0, "slope": 1, "ca": 0, "thal": 2
        },
        {
            "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
            "fbs": 0, "restecg": 1, "thalach": 155, "exang": 1,
            "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
        }
    ]
    
    print(f"Testing batch prediction for {len(patients)} patients...")
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=patients)
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    print(f"Total Patients: {result['total_patients']}")
    print(f"\nPredictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"  Patient {i}: {pred['prediction_label']} (Confidence: {pred['confidence']})")
    
    assert response.status_code == 200
    assert result['total_patients'] == 3
    print("✅ Batch prediction successful!")

def test_invalid_input():
    """Test API validation with invalid input"""
    print("\n" + "="*70)
    print("TEST 6: Invalid Input Validation")
    print("="*70)
    
    # Invalid data (age out of range)
    invalid_data = {
        "age": 200,  # Invalid age
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
    
    print("Sending invalid input (age=200)...")
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 422  # Validation error
    print("✅ Input validation working correctly!")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("HEART DISEASE PREDICTION API - TEST SUITE")
    print("="*70)
    print(f"Testing API at: {BASE_URL}")
    
    try:
        tests = [
            ("Health Check", test_health),
            ("Model Information", test_model_info),
            ("Single Prediction", test_single_prediction),
            ("Healthy Patient", test_healthy_patient),
            ("Batch Prediction", test_batch_prediction),
            ("Input Validation", test_invalid_input)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                test_func()
                passed += 1
            except Exception as e:
                print(f"❌ {test_name} failed: {str(e)}")
                failed += 1
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {len(tests)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print("="*70)
        
        if failed == 0:
            print("\n🎉 All tests passed! API is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check the logs above.")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the Docker container is running:")
        print("  docker run -p 8000:8000 heart-disease-api")
        return

if __name__ == "__main__":
    run_all_tests()
