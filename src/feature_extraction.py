"""
Feature Extraction Module for MLOps Sentiment Analysis Pipeline
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import mutual_info_classif
import pickle
import boto3
import json
from typing import Tuple, Dict, Any, Optional
import logging

# Add directories to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'config'))

from config import S3_BUCKET_NAME, AWS_REGION

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extraction class for text data using TF-IDF and feature selection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize FeatureExtractor
        
        Args:
            config: Feature engineering configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.vectorizer = None
        self.feature_selector = None
        self.feature_names = None
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        logger.info("FeatureExtractor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration"""
        return {
            'vectorizer_type': 'tfidf',
            'max_features': 1000,  # Reduced for small datasets
            'ngram_range': (1, 2),
            'min_df': 1,  # Reduced to allow more features in small datasets
            'max_df': 0.95,
            'stop_words': 'english',
            'lowercase': True,
            'strip_accents': 'unicode',
            'token_pattern': r'\b\w+\b',
            'use_chi2': True,
            'chi2_k': 50,  # Much smaller for small datasets
            'use_mutual_info': False,
            'mutual_info_k': 50  # Much smaller for small datasets
        }
    
    def create_vectorizer(self) -> TfidfVectorizer:
        """
        Create and configure TF-IDF vectorizer
        
        Returns:
            Configured TfidfVectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=self.config['ngram_range'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df'],
            stop_words=self.config['stop_words'],
            lowercase=self.config['lowercase'],
            strip_accents=self.config['strip_accents'],
            token_pattern=self.config['token_pattern']
        )
        
        logger.info(f"Created TF-IDF vectorizer with {self.config['max_features']} max features")
        return vectorizer
    
    def create_feature_selector(self, method: str = 'chi2', n_features: int = None) -> SelectKBest:
        """
        Create feature selector
        
        Args:
            method: Selection method ('chi2' or 'mutual_info')
            n_features: Number of available features (for adaptive k selection)
            
        Returns:
            Configured SelectKBest selector
        """
        if method == 'chi2':
            k = self.config['chi2_k']
        elif method == 'mutual_info':
            k = self.config['mutual_info_k']
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Adaptive k selection: use min of configured k and available features
        if n_features is not None:
            k = min(k, n_features)
            logger.info(f"Adaptive k selection: using {k} features (available: {n_features}, configured: {self.config.get('chi2_k' if method == 'chi2' else 'mutual_info_k')})")
        
        if method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
        else:  # mutual_info
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        logger.info(f"Created feature selector using {method} with k={k}")
        return selector
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit vectorizer and feature selector, then transform data
        
        Args:
            X: Text data (pandas Series)
            y: Target labels (pandas Series)
            
        Returns:
            Tuple of (feature_matrix, selected_features)
        """
        try:
            logger.info(f"Starting feature extraction for {len(X)} samples")
            
            # Create and fit vectorizer
            self.vectorizer = self.create_vectorizer()
            X_vectorized = self.vectorizer.fit_transform(X)
            
            # Get feature names
            self.feature_names = self.vectorizer.get_feature_names_out()
            logger.info(f"Vectorized to {X_vectorized.shape[1]} features")
            
            # Apply feature selection if enabled
            if self.config['use_chi2'] or self.config['use_mutual_info']:
                method = 'chi2' if self.config['use_chi2'] else 'mutual_info'
                # Pass the number of available features for adaptive k selection
                self.feature_selector = self.create_feature_selector(method, n_features=X_vectorized.shape[1])
                X_selected = self.feature_selector.fit_transform(X_vectorized, y)
                
                # Update feature names
                selected_indices = self.feature_selector.get_support(indices=True)
                self.feature_names = self.feature_names[selected_indices]
                
                logger.info(f"Selected {X_selected.shape[1]} features using {method}")
                return X_selected, self.feature_names
            else:
                logger.info("No feature selection applied")
                return X_vectorized.toarray(), self.feature_names
                
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            raise
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Transform new data using fitted vectorizer and selector
        
        Args:
            X: Text data (pandas Series)
            
        Returns:
            Transformed feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        try:
            # Transform using fitted vectorizer
            X_vectorized = self.vectorizer.transform(X)
            
            # Apply feature selection if enabled
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X_vectorized)
                return X_selected
            else:
                return X_vectorized.toarray()
                
        except Exception as e:
            logger.error(f"Error in transform: {e}")
            raise
    
    def get_feature_names(self) -> np.ndarray:
        """
        Get feature names after fitting
        
        Returns:
            Array of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature extractor not fitted yet")
        return self.feature_names
    
    def save_to_s3(self, filename: str = "feature_extractor.pkl") -> bool:
        """
        Save fitted vectorizer and selector to S3
        
        Args:
            filename: Name of the file to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vectorizer is None:
                raise ValueError("No fitted vectorizer to save")
            
            # Prepare data to save
            save_data = {
                'vectorizer': self.vectorizer,
                'feature_selector': self.feature_selector,
                'feature_names': self.feature_names,
                'config': self.config
            }
            
            # Serialize to pickle
            pickle_data = pickle.dumps(save_data)
            
            # Upload to S3
            key = f"artifacts/{filename}"
            self.s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=key,
                Body=pickle_data,
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Feature extractor saved to S3: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to S3: {e}")
            return False
    
    def load_from_s3(self, filename: str = "feature_extractor.pkl") -> bool:
        """
        Load fitted vectorizer and selector from S3
        
        Args:
            filename: Name of the file to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Download from S3
            key = f"artifacts/{filename}"
            response = self.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            pickle_data = response['Body'].read()
            
            # Deserialize
            save_data = pickle.loads(pickle_data)
            
            # Restore components
            self.vectorizer = save_data['vectorizer']
            self.feature_selector = save_data['feature_selector']
            self.feature_names = save_data['feature_names']
            self.config = save_data['config']
            
            logger.info(f"Feature extractor loaded from S3: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return False
    
    def get_feature_importance(self, y: pd.Series) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if self.feature_selector is None:
            raise ValueError("No feature selector fitted")
        
        try:
            scores = self.feature_selector.scores_
            feature_names = self.get_feature_names()
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, scores))
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            logger.info(f"Generated feature importance for {len(sorted_features)} features")
            return dict(sorted_features)
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}


def feature_extraction(df: pd.DataFrame, 
                      text_column: str = 'cleaned_text',
                      target_column: str = 'sentiment',
                      config: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, FeatureExtractor]:
    """
    Convenience function for feature extraction
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        target_column: Name of target column
        config: Feature engineering configuration
        
    Returns:
        Tuple of (feature_matrix, feature_names, fitted_extractor)
    """
    try:
        # Initialize extractor
        extractor = FeatureExtractor(config)
        
        # Prepare data
        X = df[text_column]
        y = df[target_column]
        
        # Fit and transform
        feature_matrix, feature_names = extractor.fit_transform(X, y)
        
        logger.info(f"Feature extraction completed: {feature_matrix.shape}")
        
        return feature_matrix, feature_names, extractor
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        raise


# Test the module
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'cleaned_text': [
            "love product amazing quality",
            "terrible quality waste money",
            "good value price excellent",
            "worst experience ever bad"
        ],
        'sentiment': [1, 0, 1, 0]
    })
    
    # Test feature extraction
    X, feature_names, extractor = feature_extraction(sample_data)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Sample features: {feature_names[:10]}")
    
    # Test saving
    success = extractor.save_to_s3("test_feature_extractor.pkl")
    print(f"Save to S3: {'Success' if success else 'Failed'}")
