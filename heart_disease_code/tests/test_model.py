"""Unit tests for model training and evaluation"""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.training import (
    split_data, train_logistic_regression, train_random_forest,
    evaluate_model, cross_validate_model, save_model, load_model
)
import os


@pytest.fixture
def sample_classification_data():
    """Create sample classification data"""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y)
    return X_df, y_series


def test_split_data(sample_classification_data):
    """Test data splitting"""
    X, y = sample_classification_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_train_logistic_regression(sample_classification_data):
    """Test logistic regression training"""
    X, y = sample_classification_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_logistic_regression(X_train, y_train)
    
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert model.classes_.tolist() == [0, 1]


def test_train_random_forest(sample_classification_data):
    """Test random forest training"""
    X, y = sample_classification_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_random_forest(X_train, y_train, n_estimators=50)
    
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert model.n_estimators == 50


def test_evaluate_model(sample_classification_data):
    """Test model evaluation"""
    X, y = sample_classification_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_cross_validate_model(sample_classification_data):
    """Test cross-validation"""
    X, y = sample_classification_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_logistic_regression(X_train, y_train)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    
    assert 'cv_scores' in cv_results
    assert 'cv_mean' in cv_results
    assert 'cv_std' in cv_results
    assert len(cv_results['cv_scores']) == 5


def test_save_and_load_model(sample_classification_data, tmp_path):
    """Test model saving and loading"""
    X, y = sample_classification_data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_logistic_regression(X_train, y_train)
    
    model_path = tmp_path / "test_model.pkl"
    save_model(model, str(model_path))
    
    assert os.path.exists(model_path)
    
    loaded_model = load_model(str(model_path))
    
    original_pred = model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    
    assert np.array_equal(original_pred, loaded_pred)
