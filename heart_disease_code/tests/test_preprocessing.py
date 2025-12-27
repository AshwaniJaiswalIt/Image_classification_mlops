"""Unit tests for preprocessing functions"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    clean_missing_values, impute_missing_values,
    convert_target_to_binary, create_onehot_encoding,
    prepare_features_target, scale_features
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'age': [50, 60, 45],
        'sex': [1, 0, 1],
        'cp': [2, 1, 3],
        'trestbps': [120, 130, 140],
        'chol': [200, 210, 220],
        'fbs': [0, 1, 0],
        'restecg': [0, 1, 0],
        'thalach': [150, 160, 155],
        'exang': [0, 1, 0],
        'oldpeak': [1.0, 2.0, 1.5],
        'slope': [1, 2, 1],
        'ca': [0, 1, '?'],
        'thal': [3, 3, 3],
        'target': [0, 1, 2]
    }
    return pd.DataFrame(data)


def test_clean_missing_values(sample_data):
    """Test cleaning missing values"""
    df_clean = clean_missing_values(sample_data)
    
    assert df_clean['ca'].isna().sum() == 1
    assert df_clean.dtypes['ca'] == np.float64


def test_impute_missing_values(sample_data):
    """Test imputation of missing values"""
    df_clean = clean_missing_values(sample_data)
    df_imputed, imputer = impute_missing_values(df_clean, columns_to_impute=['ca'])
    
    assert df_imputed['ca'].isna().sum() == 0
    assert imputer is not None


def test_convert_target_to_binary(sample_data):
    """Test target conversion to binary"""
    df_binary = convert_target_to_binary(sample_data)
    
    assert df_binary['target'].nunique() == 2
    assert set(df_binary['target'].unique()) == {0, 1}


def test_create_onehot_encoding(sample_data):
    """Test one-hot encoding"""
    df_clean = clean_missing_values(sample_data)
    df_imputed, _ = impute_missing_values(df_clean)
    df_encoded = create_onehot_encoding(df_imputed, columns_to_encode=['ca', 'thal'])
    
    # With drop_first=True, we might have same or more columns depending on unique values
    # Check that original columns are removed
    assert 'ca' not in df_encoded.columns
    assert 'thal' not in df_encoded.columns
    # Check that new encoded columns exist
    assert any('ca_' in col for col in df_encoded.columns)


def test_prepare_features_target(sample_data):
    """Test feature-target separation"""
    X, y = prepare_features_target(sample_data)
    
    assert X.shape[1] == sample_data.shape[1] - 1
    assert len(y) == len(sample_data)
    assert 'target' not in X.columns


def test_scale_features():
    """Test feature scaling"""
    X_train = pd.DataFrame({
        'age': [50, 60, 45],
        'chol': [200, 210, 220]
    })
    X_test = pd.DataFrame({
        'age': [55],
        'chol': [205]
    })
    
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, ['age', 'chol']
    )
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert scaler is not None
    assert abs(X_train_scaled['age'].mean()) < 0.01  # Should be near 0
