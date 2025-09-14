"""
Data Loading Module for MLOps Sentiment Analysis Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import logging
import os
from pathlib import Path

# Import existing modules
from data_ingestion import DataIngestion
from etl_pipeline import ETLPipeline
from training_config import DATA_CONFIG, DEFAULT_CONFIG
from config.config import S3_BUCKET_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading and preprocessing class for ML training
    """
    
    def __init__(self, s3_bucket: str = None, config: Dict[str, Any] = None):
        """
        Initialize DataLoader
        
        Args:
            s3_bucket: S3 bucket name
            config: Data configuration dictionary
        """
        self.s3_bucket = s3_bucket or S3_BUCKET_NAME
        self.config = config or DATA_CONFIG
        
        # Initialize data ingestion and ETL
        self.data_ingestion = DataIngestion(self.s3_bucket)
        self.etl_pipeline = ETLPipeline(self.s3_bucket)
        
        logger.info(f"DataLoader initialized with bucket: {self.s3_bucket}")
    
    def load_training_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load processed training data from S3
        
        Args:
            data_path: Path to processed data in S3
            
        Returns:
            Loaded DataFrame
        """
        try:
            data_path = data_path or f"{self.config['processed_data_path']}processed_sentiment_data.json"
            
            logger.info(f"Loading data from S3: {data_path}")
            
            # Load data from S3
            df = self.data_ingestion.download_raw_data(data_path.replace('processed-data/', ''))
            
            if df is None:
                raise ValueError(f"Failed to load data from {data_path}")
            
            logger.info(f"Loaded {len(df)} samples from S3")
            
            # Validate data quality
            self.validate_data_quality(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def load_raw_data_and_process(self, raw_data_path: str = None) -> pd.DataFrame:
        """
        Load raw data from S3 and process it through ETL pipeline
        
        Args:
            raw_data_path: Path to raw data in S3
            
        Returns:
            Processed DataFrame
        """
        try:
            raw_data_path = raw_data_path or f"{self.config['raw_data_path']}sample_sentiment_data.json"
            
            logger.info(f"Loading raw data from S3: {raw_data_path}")
            
            # Load raw data
            raw_df = self.data_ingestion.download_raw_data(raw_data_path.replace('raw-data/', ''))
            
            if raw_df is None:
                raise ValueError(f"Failed to load raw data from {raw_data_path}")
            
            logger.info(f"Loaded {len(raw_df)} raw samples")
            
            # Process through ETL pipeline
            processed_df = self.etl_pipeline.process_data(raw_df)
            
            # Upload processed data
            processed_filename = f"processed_{raw_data_path.split('/')[-1]}"
            self.etl_pipeline.upload_processed_data(processed_df, processed_filename)
            
            logger.info(f"Processed and uploaded {len(processed_df)} samples")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error loading and processing data: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame, 
                   test_size: float = None, 
                   validation_size: float = None,
                   random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        try:
            test_size = test_size or self.config['test_size']
            validation_size = validation_size or self.config['validation_size']
            random_state = random_state or self.config['random_state']
            
            logger.info(f"Splitting data - Test: {test_size}, Validation: {validation_size}")
            
            # Prepare features and labels
            X, y = self.prepare_features_labels(df)
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y
            )
            
            # Second split: separate validation set from remaining data
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_temp
            )
            
            # Create DataFrames
            train_df = pd.DataFrame({
                'text': X_train,
                'sentiment': y_train
            })
            
            val_df = pd.DataFrame({
                'text': X_val,
                'sentiment': y_val
            })
            
            test_df = pd.DataFrame({
                'text': X_test,
                'sentiment': y_test
            })
            
            logger.info(f"Data split completed:")
            logger.info(f"  Train: {len(train_df)} samples")
            logger.info(f"  Validation: {len(val_df)} samples")
            logger.info(f"  Test: {len(test_df)} samples")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    def prepare_features_labels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare features (X) and labels (y) for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            # Check if cleaned_text column exists
            if 'cleaned_text' in df.columns:
                features = df['cleaned_text']
            elif 'text' in df.columns:
                features = df['text']
            else:
                raise ValueError("No text column found in DataFrame")
            
            # Get labels
            if 'sentiment' not in df.columns:
                raise ValueError("No sentiment column found in DataFrame")
            
            labels = df['sentiment']
            
            # Remove any NaN values
            mask = features.notna() & labels.notna()
            features = features[mask]
            labels = labels[mask]
            
            logger.info(f"Prepared {len(features)} features and labels")
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Error preparing features and labels: {e}")
            raise
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if data quality is acceptable
        """
        try:
            logger.info("Validating data quality...")
            
            # Check required columns
            required_columns = self.config['required_columns']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(df['sentiment']):
                raise ValueError("Sentiment column must be numeric")
            
            # Check sentiment values
            valid_sentiments = set(self.config['sentiment_values'])
            actual_sentiments = set(df['sentiment'].unique())
            
            if not actual_sentiments.issubset(valid_sentiments):
                invalid_sentiments = actual_sentiments - valid_sentiments
                raise ValueError(f"Invalid sentiment values: {invalid_sentiments}")
            
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            max_missing = self.config['max_missing_ratio']
            
            if missing_ratio > max_missing:
                raise ValueError(f"Too many missing values: {missing_ratio:.2%} > {max_missing:.2%}")
            
            # Check text length
            if 'text' in df.columns:
                text_lengths = df['text'].str.len()
                min_length = self.config['min_text_length']
                max_length = self.config['max_text_length']
                
                short_texts = (text_lengths < min_length).sum()
                long_texts = (text_lengths > max_length).sum()
                
                if short_texts > 0:
                    logger.warning(f"{short_texts} texts shorter than {min_length} characters")
                
                if long_texts > 0:
                    logger.warning(f"{long_texts} texts longer than {max_length} characters")
            
            # Check class balance
            if 'sentiment' in df.columns:
                class_counts = df['sentiment'].value_counts()
                min_samples = self.config['min_samples_per_class']
                
                for sentiment, count in class_counts.items():
                    if count < min_samples:
                        logger.warning(f"Class {sentiment} has only {count} samples (min: {min_samples})")
            
            logger.info("✅ Data quality validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data quality validation failed: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data summary
        """
        try:
            summary = {
                'total_samples': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Sentiment distribution
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts().to_dict()
                summary['sentiment_distribution'] = sentiment_counts
                summary['class_balance'] = {
                    sentiment: count / len(df) 
                    for sentiment, count in sentiment_counts.items()
                }
            
            # Text length statistics
            if 'text' in df.columns:
                text_lengths = df['text'].str.len()
                summary['text_length_stats'] = {
                    'mean': text_lengths.mean(),
                    'std': text_lengths.std(),
                    'min': text_lengths.min(),
                    'max': text_lengths.max(),
                    'median': text_lengths.median()
                }
            
            # Cleaned text length statistics
            if 'cleaned_text' in df.columns:
                cleaned_lengths = df['cleaned_text'].str.len()
                summary['cleaned_text_length_stats'] = {
                    'mean': cleaned_lengths.mean(),
                    'std': cleaned_lengths.std(),
                    'min': cleaned_lengths.min(),
                    'max': cleaned_lengths.max(),
                    'median': cleaned_lengths.median()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            return {}
    
    def save_data_splits(self, train_df: pd.DataFrame, 
                        val_df: pd.DataFrame, 
                        test_df: pd.DataFrame,
                        prefix: str = "split") -> Dict[str, str]:
        """
        Save data splits to S3
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            prefix: Prefix for filenames
            
        Returns:
            Dictionary with saved file paths
        """
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Save training data
            train_filename = f"{prefix}_train_{timestamp}.json"
            train_success = self.data_ingestion.upload_raw_data(
                train_df, train_filename
            )
            
            # Save validation data
            val_filename = f"{prefix}_val_{timestamp}.json"
            val_success = self.data_ingestion.upload_raw_data(
                val_df, val_filename
            )
            
            # Save test data
            test_filename = f"{prefix}_test_{timestamp}.json"
            test_success = self.data_ingestion.upload_raw_data(
                test_df, test_filename
            )
            
            if not all([train_success, val_success, test_success]):
                raise ValueError("Failed to save some data splits")
            
            saved_files = {
                'train': f"{self.config['processed_data_path']}{train_filename}",
                'validation': f"{self.config['processed_data_path']}{val_filename}",
                'test': f"{self.config['processed_data_path']}{test_filename}"
            }
            
            logger.info(f"Data splits saved to S3: {saved_files}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving data splits: {e}")
            raise


def load_training_data(data_path: str = None) -> pd.DataFrame:
    """
    Convenience function to load training data
    
    Args:
        data_path: Path to data in S3
        
    Returns:
        Loaded DataFrame
    """
    loader = DataLoader()
    return loader.load_training_data(data_path)


def split_data(df: pd.DataFrame, 
               test_size: float = None, 
               validation_size: float = None,
               random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to split data
    
    Args:
        df: Input DataFrame
        test_size: Proportion for testing
        validation_size: Proportion for validation
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = DataLoader()
    return loader.split_data(df, test_size, validation_size, random_state)


def prepare_features_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Convenience function to prepare features and labels
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (features, labels)
    """
    loader = DataLoader()
    return loader.prepare_features_labels(df)


def validate_data_quality(df: pd.DataFrame) -> bool:
    """
    Convenience function to validate data quality
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if data quality is acceptable
    """
    loader = DataLoader()
    return loader.validate_data_quality(df)
