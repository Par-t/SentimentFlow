import sys
from pathlib import Path

# Add directories to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'config'))

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import boto3
from datetime import datetime
from charset_normalizer import from_bytes
from typing import Optional, Dict, Any

from data_ingestion import DataIngestion
from config import S3_BUCKET_NAME, AWS_REGION, DATA_DIR
from feature_extraction import feature_extraction, FeatureExtractor

class ETLPipeline:
    def __init__(self, bucket_name=None):
        self.bucket_name = bucket_name or S3_BUCKET_NAME
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Dynamic label mapping - will be created based on unique values in dataset
        self.label_mapping = {}
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using charset-normalizer library"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                
                # Use charset-normalizer to detect encoding
                detected = from_bytes(raw_data).best()
                
                if detected is not None:
                    encoding = detected.encoding
                    confidence = detected.encoding_confidence
                    
                    print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                    
                    # Return detected encoding or fallback to utf-8
                    return encoding if confidence > 0.7 else 'utf-8'
                else:
                    print("No encoding detected. Using utf-8 as fallback.")
                    return 'utf-8'
                
        except Exception as e:
            print(f"Encoding detection failed: {e}. Using utf-8 as fallback.")
            return 'utf-8'
    
    def apply_label_mapping(self, df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """Apply dynamic label mapping to label column based on unique values"""
        print(f"Applying dynamic label mapping to column: {label_column}")
        
        # Get unique values and create mapping
        unique_labels = df[label_column].dropna().unique()
        unique_labels = sorted(unique_labels)  # Sort for consistent mapping
        
        print(f"Found {len(unique_labels)} unique labels:")
        
        # Create mapping from labels to numbers (0, 1, 2, ...)
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Print the mapping
        for label, numeric_value in self.label_mapping.items():
            print(f"  '{label}' -> {numeric_value}")
        
        # Apply mapping
        df[label_column] = df[label_column].map(self.label_mapping)
        
        # Handle any unmapped values (shouldn't happen, but just in case)
        unmapped_count = df[label_column].isna().sum()
        if unmapped_count > 0:
            print(f"Warning: {unmapped_count} unmapped labels found.")
            # Assign the highest number + 1 to unmapped values
            max_value = df[label_column].max() if not df[label_column].isna().all() else -1
            df[label_column] = df[label_column].fillna(max_value + 1)
        
        # Convert to int
        df[label_column] = df[label_column].astype(int)
        
        # Print final label distribution
        label_counts = df[label_column].value_counts().sort_index()
        print(f"\nFinal label distribution:")
        for numeric_label, count in label_counts.items():
            # Find original label name
            original_label = next((k for k, v in self.label_mapping.items() if v == numeric_label), f"Unknown_{numeric_label}")
            print(f"  {numeric_label} ({original_label}): {count} samples")
        
        return df
    
    def save_label_mapping(self, filename: str, tracker=None, dataset_name=None) -> bool:
        """Save label mapping to S3 for reproducibility"""
        try:
            mapping_json = json.dumps(self.label_mapping, indent=2)
            key = f"label-mappings/{filename}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=mapping_json,
                ContentType='application/json'
            )
            print(f"Label mapping saved to S3: {filename}")
            # Track the label mapping
            if tracker and dataset_name:
                tracker.add_artifact(dataset_name, "label_mapping", filename)
            return True
        except Exception as e:
            print(f"Error saving label mapping: {e}")
            return False
    
    def load_label_mapping(self, filename: str) -> bool:
        """Load label mapping from S3"""
        try:
            key = f"label-mappings/{filename}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            mapping_json = response['Body'].read().decode('utf-8')
            self.label_mapping = json.loads(mapping_json)
            print(f"Label mapping loaded from S3: {filename}")
            return True
        except Exception as e:
            print(f"Error loading label mapping: {e}")
            return False
    
    def apply_cleaning_functions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning functions"""
        print("Applying data cleaning functions...")
        
        original_count = len(df)
        
        # Drop rows with missing values in key columns
        df = df.dropna(subset=['text', 'sentiment'])
        after_dropna = len(df)
        print(f"Dropped {original_count - after_dropna} rows with missing values")
        
        # Convert text to lowercase
        df['text'] = df['text'].astype(str).str.lower()
        print("Converted text to lowercase")
        
        # Remove completely empty text rows
        df = df[df['text'].str.strip() != '']
        after_empty_removal = len(df)
        print(f"Removed {after_dropna - after_empty_removal} rows with empty text")
        
        print(f"Cleaning completed. Final dataset: {len(df)} rows")
        return df
    
    def process_data(self, df, label_column: Optional[str] = None, tracker=None, dataset_name=None):
        """Process the entire dataset with enhanced cleaning and label mapping"""
        print("Starting enhanced data processing...")
        
        # Apply cleaning functions
        df = self.apply_cleaning_functions(df)
        
        # Apply label mapping if label_column is specified
        if label_column and label_column in df.columns:
            df = self.apply_label_mapping(df, label_column)
            
            # Save label mapping for reproducibility with dataset name
            if dataset_name:
                dataset_base = dataset_name.replace('.csv', '').replace('.json', '')
                mapping_filename = f"label_mapping_{dataset_base}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                mapping_filename = f"label_mapping_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_label_mapping(mapping_filename, tracker, dataset_name)
        
        # Clean text using existing method
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Add metadata
        df['processed_at'] = datetime.now()
        df['text_length'] = df['cleaned_text'].apply(len)
        
        # Remove rows with empty cleaned text
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"Enhanced processing completed. Final dataset: {len(df)} samples")
        return df
    
    def feature_extraction(self, df, config=None, tracker=None, dataset_name=None):
        """
        Extract features from processed text data
        
        Args:
            df: Processed DataFrame with cleaned_text column
            config: Feature engineering configuration
            
        Returns:
            Tuple of (feature_matrix, feature_names, fitted_extractor)
        """
        try:
            print("Starting feature extraction...")
            
            # Check if cleaned_text column exists
            if 'cleaned_text' not in df.columns:
                raise ValueError("cleaned_text column not found. Run process_data first.")
            
            if 'sentiment' not in df.columns:
                raise ValueError("sentiment column not found.")
            
            # Extract features
            feature_matrix, feature_names, extractor = feature_extraction(
                df, 
                text_column='cleaned_text',
                target_column='sentiment',
                config=config
            )
            
            print(f"Feature extraction completed: {feature_matrix.shape}")
            print(f"Number of features: {len(feature_names)}")
            
            # Save feature extractor to S3 with dataset name
            if dataset_name:
                dataset_base = dataset_name.replace('.csv', '').replace('.json', '')
                extractor_filename = f"feature_extractor_{dataset_base}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            else:
                extractor_filename = f"feature_extractor_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            save_success = extractor.save_to_s3(extractor_filename)
            
            if save_success:
                print(f"Feature extractor saved to S3: {extractor_filename}")
                # Track the feature extractor
                if tracker and dataset_name:
                    tracker.add_artifact(dataset_name, "feature_extractor", extractor_filename)
            else:
                print("Warning: Failed to save feature extractor to S3")
            
            return feature_matrix, feature_names, extractor
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            raise
    
    def load_csv_with_encoding_detection(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with automatic encoding detection"""
        print(f"Loading CSV file: {file_path}")
        
        # Detect encoding
        encoding = self.detect_encoding(file_path)
        
        try:
            # Try to load with detected encoding
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded CSV with encoding: {encoding}")
            return df
            
        except UnicodeDecodeError:
            print(f"Failed to load with {encoding}. Trying utf-8...")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                print("Successfully loaded CSV with utf-8 encoding")
                return df
            except UnicodeDecodeError:
                print("Failed to load with utf-8. Trying latin-1...")
                df = pd.read_csv(file_path, encoding='latin-1')
                print("Successfully loaded CSV with latin-1 encoding")
                return df
    
    def upload_processed_data(self, df, filename):
        """Upload processed data to S3"""
        try:
            data_json = df.to_json(orient='records')
            key = f"processed-data/{filename}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data_json,
                ContentType='application/json'
            )
            print(f"Successfully uploaded processed data: {filename}")
            return True
        except Exception as e:
            print(f"Error uploading processed data: {e}")
            return False

# Test the ETL pipeline
if __name__ == "__main__":
    # Download raw data
    ingestion = DataIngestion()
    raw_data = ingestion.download_raw_data("sample_sentiment_data.json")
    
    if raw_data is not None:
        # Process data
        etl = ETLPipeline()
        processed_data = etl.process_data(raw_data)
        
        # Upload processed data
        etl.upload_processed_data(processed_data, "processed_sentiment_data.json")
        
        # Extract features
        feature_matrix, feature_names, extractor = etl.feature_extraction(processed_data)
        
        print("ETL pipeline completed successfully!")
        print("Sample processed data:")
        print(processed_data[['text', 'cleaned_text', 'sentiment']].head())
        print(f"\nFeature matrix shape: {feature_matrix.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Sample features: {feature_names[:10]}")
    else:
        print("Failed to download raw data")
