
import boto3
import pandas as pd
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Add config directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'config'))

from config import S3_BUCKET_NAME, AWS_REGION

class DataIngestion:
    def __init__(self, bucket_name=None):
        self.bucket_name = bucket_name or S3_BUCKET_NAME
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    def upload_raw_data(self, data, filename):
        """Upload raw data to S3"""
        try:
            # Convert to JSON if it's a DataFrame
            if isinstance(data, pd.DataFrame):
                data_json = data.to_json(orient='records')
            else:
                data_json = json.dumps(data)
            
            # Upload to S3
            key = f"raw-data/{filename}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data_json,
                ContentType='application/json'
            )
            print(f"Successfully uploaded {filename} to S3")
            return True
        except Exception as e:
            print(f"Error uploading data: {e}")
            return False
    
    def download_raw_data(self, filename):
        """Download raw data from S3"""
        try:
            key = f"raw-data/{filename}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

# Test the class
if __name__ == "__main__":
    # Use default bucket from config
    ingestion = DataIngestion()
    
    # Test with sample data
    sample_data = {
        "text": ["I love this product!", "This is terrible.", "It's okay."],
        "sentiment": [1, 0, 1]
    }
    
    # Upload test data
    ingestion.upload_raw_data(sample_data, "test_data.json")
    
    # Download and verify
    downloaded_data = ingestion.download_raw_data("test_data.json")
    print("Downloaded data:")
    print(downloaded_data)