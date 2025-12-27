"""Data preprocessing functions for heart disease classification"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load heart disease dataset from CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset with proper column names
    """
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(filepath, header=None, names=columns)
    return df


def clean_missing_values(df):
    """Replace '?' with NaN and convert to float
    
    Args:
        df: Input dataframe
        
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    df_clean.replace("?", np.nan, inplace=True)
    df_clean = df_clean.astype(float)
    return df_clean


def impute_missing_values(df, columns_to_impute=['ca', 'thal'], strategy='median'):
    """Impute missing values using specified strategy
    
    Args:
        df: Input dataframe
        columns_to_impute: List of columns to impute
        strategy: Imputation strategy (default: 'median')
        
    Returns:
        tuple: (imputed_dataframe, fitted_imputer)
    """
    df_imputed = df.copy()
    imputer = SimpleImputer(strategy=strategy)
    df_imputed[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    return df_imputed, imputer


def convert_target_to_binary(df, target_column='target'):
    """Convert multi-class target to binary classification
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        
    Returns:
        pandas.DataFrame: Dataframe with binary target
    """
    df_binary = df.copy()
    df_binary[target_column] = (df[target_column] > 0).astype(int)
    return df_binary


def create_onehot_encoding(df, columns_to_encode=['ca', 'thal'], drop_first=True):
    """Create one-hot encoded features for specified columns
    
    Args:
        df: Input dataframe
        columns_to_encode: Columns to one-hot encode
        drop_first: Whether to drop first category (default: True)
        
    Returns:
        pandas.DataFrame: One-hot encoded dataframe
    """
    df_encoded = pd.get_dummies(
        df, 
        columns=columns_to_encode, 
        prefix=columns_to_encode,
        drop_first=drop_first, 
        dtype=float
    )
    return df_encoded


def prepare_features_target(df, target_column='target'):
    """Separate features and target variable
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        
    Returns:
        tuple: (X, y) features and target
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def scale_features(X_train, X_test, numerical_features):
    """Scale numerical features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
        numerical_features: List of numerical feature names
        
    Returns:
        tuple: (scaled_X_train, scaled_X_test, fitted_scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    return X_train_scaled, X_test_scaled, scaler
