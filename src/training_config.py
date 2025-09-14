"""
Training Configuration Module for MLOps Sentiment Analysis Pipeline
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import os
from config.config import S3_BUCKET_NAME, AWS_REGION, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE


@dataclass
class TrainingConfig:
    """Configuration class for model training"""
    # Data configuration
    s3_bucket: str
    raw_data_path: str
    processed_data_path: str
    models_path: str
    artifacts_path: str
    
    # Data splitting parameters
    test_size: float
    validation_size: float
    random_state: int
    
    # MLflow configuration
    experiment_name: str
    tracking_uri: str
    
    # Feature engineering parameters
    max_features: int
    ngram_range: Tuple[int, int]
    min_df: int
    max_df: float
    
    # Model training parameters
    cv_folds: int
    scoring_metric: str
    
    # Performance thresholds
    min_accuracy_threshold: float
    min_f1_threshold: float


def create_training_config(
    s3_bucket: str = None,
    experiment_name: str = None,
    tracking_uri: str = None,
    max_features: int = None,
    **kwargs
) -> TrainingConfig:
    """
    Create training configuration with optional overrides
    
    Args:
        s3_bucket: S3 bucket name (default: from config)
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI
        max_features: Maximum number of features for TF-IDF
        **kwargs: Additional configuration overrides
    
    Returns:
        TrainingConfig object
    """
    return TrainingConfig(
        # Data configuration
        s3_bucket=s3_bucket or S3_BUCKET_NAME,
        raw_data_path=kwargs.get('raw_data_path', 'raw-data/'),
        processed_data_path=kwargs.get('processed_data_path', 'processed-data/'),
        models_path=kwargs.get('models_path', 'models/'),
        artifacts_path=kwargs.get('artifacts_path', 'artifacts/'),
        
        # Data splitting
        test_size=kwargs.get('test_size', TEST_SIZE),
        validation_size=kwargs.get('validation_size', VALIDATION_SIZE),
        random_state=kwargs.get('random_state', RANDOM_STATE),
        
        # MLflow configuration
        experiment_name=experiment_name or 'sentiment_analysis_experiments',
        tracking_uri=tracking_uri or 'file:./mlruns',
        
        # Feature engineering
        max_features=max_features or 5000,
        ngram_range=kwargs.get('ngram_range', (1, 2)),
        min_df=kwargs.get('min_df', 2),
        max_df=kwargs.get('max_df', 0.95),
        
        # Model training
        cv_folds=kwargs.get('cv_folds', 5),
        scoring_metric=kwargs.get('scoring_metric', 'f1'),
        
        # Performance thresholds
        min_accuracy_threshold=kwargs.get('min_accuracy_threshold', 0.8),
        min_f1_threshold=kwargs.get('min_f1_threshold', 0.8)
    )


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get model configurations for different algorithms
    
    Returns:
        Dictionary of model configurations
    """
    return {
        'logistic_regression': {
            'model_class': 'LogisticRegression',
            'parameters': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000],
                'random_state': [42],
                'solver': ['liblinear', 'lbfgs']
            },
            'default_params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'liblinear'
            }
        },
        
        'random_forest': {
            'model_class': 'RandomForestClassifier',
            'parameters': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            },
            'default_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        },
        
        'svm': {
            'model_class': 'SVC',
            'parameters': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'random_state': [42]
            },
            'default_params': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
        },
        
        'naive_bayes': {
            'model_class': 'MultinomialNB',
            'parameters': {
                'alpha': [0.1, 1.0, 10.0],
                'fit_prior': [True, False]
            },
            'default_params': {
                'alpha': 1.0,
                'fit_prior': True
            }
        },
        
        'gradient_boosting': {
            'model_class': 'GradientBoostingClassifier',
            'parameters': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'random_state': [42]
            },
            'default_params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        }
    }


def get_data_config() -> Dict[str, Any]:
    """
    Get data loading and splitting configuration
    
    Returns:
        Data configuration dictionary
    """
    return {
        's3_bucket': S3_BUCKET_NAME,
        'raw_data_path': 'raw-data/',
        'processed_data_path': 'processed-data/',
        'models_path': 'models/',
        'artifacts_path': 'artifacts/',
        
        # Data splitting
        'test_size': TEST_SIZE,
        'validation_size': VALIDATION_SIZE,
        'random_state': RANDOM_STATE,
        
        # Data quality checks
        'min_text_length': 10,
        'max_text_length': 1000,
        'required_columns': ['text', 'sentiment', 'cleaned_text'],
        
        # Data validation
        'sentiment_values': [0, 1],  # Binary classification
        'max_missing_ratio': 0.05,  # Max 5% missing values
        'min_samples_per_class': 10  # Min samples per sentiment class
    }


def get_mlflow_config() -> Dict[str, Any]:
    """
    Get MLflow tracking configuration
    
    Returns:
        MLflow configuration dictionary
    """
    return {
        'tracking_uri': 'file:./mlruns',
        'experiment_name': 'sentiment_analysis_experiments',
        'artifact_location': './mlruns/artifacts',
        
        # Experiment tags
        'default_tags': {
            'project': 'sentiment_analysis',
            'version': '1.0',
            'environment': 'development'
        },
        
        # Model registry
        'model_registry_name': 'sentiment_analysis_models',
        'staging_stage': 'Staging',
        'production_stage': 'Production',
        'archived_stage': 'Archived',
        
        # Logging configuration
        'log_models': True,
        'log_artifacts': True,
        'log_parameters': True,
        'log_metrics': True
    }


def get_feature_engineering_config() -> Dict[str, Any]:
    """
    Get feature engineering configuration
    
    Returns:
        Feature engineering configuration dictionary
    """
    return {
        'vectorizer_type': 'tfidf',
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'stop_words': 'english',
        'lowercase': True,
        'strip_accents': 'unicode',
        'token_pattern': r'\b\w+\b',
        
        # Text preprocessing
        'remove_special_chars': True,
        'remove_numbers': True,
        'lemmatize': True,
        'remove_stopwords': True,
        
        # Feature selection
        'use_chi2': True,
        'chi2_k': 1000,
        'use_mutual_info': False,
        'mutual_info_k': 1000
    }


def get_evaluation_config() -> Dict[str, Any]:
    """
    Get model evaluation configuration
    
    Returns:
        Evaluation configuration dictionary
    """
    return {
        'metrics': [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'confusion_matrix'
        ],
        
        'cv_folds': 5,
        'scoring': 'f1',
        'return_train_score': True,
        
        # Performance thresholds
        'min_accuracy': 0.8,
        'min_f1': 0.8,
        'min_precision': 0.75,
        'min_recall': 0.75,
        
        # Cross-validation strategy
        'cv_strategy': 'stratified',
        'shuffle': True,
        'random_state': RANDOM_STATE
    }


def get_hyperparameter_tuning_config() -> Dict[str, Any]:
    """
    Get hyperparameter tuning configuration
    
    Returns:
        Hyperparameter tuning configuration dictionary
    """
    return {
        'tuning_method': 'grid_search',  # or 'random_search', 'bayesian'
        'cv_folds': 3,
        'n_jobs': -1,
        'scoring': 'f1',
        'verbose': 1,
        
        # Grid search specific
        'grid_search_params': {
            'cv': 3,
            'scoring': 'f1',
            'n_jobs': -1,
            'verbose': 1
        },
        
        # Random search specific
        'random_search_params': {
            'n_iter': 50,
            'cv': 3,
            'scoring': 'f1',
            'n_jobs': -1,
            'verbose': 1,
            'random_state': RANDOM_STATE
        }
    }


# Default configuration instance
DEFAULT_CONFIG = create_training_config()

# Export commonly used configurations
MODEL_CONFIGS = get_model_configs()
DATA_CONFIG = get_data_config()
MLFLOW_CONFIG = get_mlflow_config()
FEATURE_CONFIG = get_feature_engineering_config()
EVALUATION_CONFIG = get_evaluation_config()
TUNING_CONFIG = get_hyperparameter_tuning_config()
