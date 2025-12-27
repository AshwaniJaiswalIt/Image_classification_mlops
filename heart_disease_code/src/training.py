"""Model training and evaluation functions"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pickle


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_logistic_regression(X_train, y_train, max_iter=2000, random_state=42):
    """Train Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        max_iter: Maximum iterations (default: 2000)
        random_state: Random seed (default: 42)
        
    Returns:
        trained model
    """
    model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees (default: 200)
        random_state: Random seed (default: 42)
        
    Returns:
        trained model
    """
    model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics


def cross_validate_model(model, X_train, y_train, cv=5):
    """Perform cross-validation
    
    Args:
        model: Model to validate
        X_train: Training features
        y_train: Training target
        cv: Number of folds (default: 5)
        
    Returns:
        dict: CV scores and statistics
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    return {
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


def save_model(model, filepath):
    """Save model to pickle file
    
    Args:
        model: Model to save
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load model from pickle file
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
