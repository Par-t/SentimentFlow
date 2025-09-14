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

from data_ingestion import DataIngestion
from config.config import S3_BUCKET_NAME, AWS_REGION, DATA_DIR

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
    
    def process_data(self, df):
        """Process the entire dataset"""
        print("Starting data processing...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Add metadata
        df['processed_at'] = datetime.now()
        df['text_length'] = df['cleaned_text'].apply(len)
        
        # Remove rows with empty cleaned text
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"Processed {len(df)} samples")
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
        
        print("ETL pipeline completed successfully!")
        print("Sample processed data:")
        print(processed_data[['text', 'cleaned_text', 'sentiment']].head())
    else:
        print("Failed to download raw data")
