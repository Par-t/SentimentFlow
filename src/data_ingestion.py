
import boto3
import pandas as pd
import json
from datetime import datetime
import os
import sys
from pathlib import Path
from io import StringIO
from charset_normalizer import from_bytes

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
    
    def detect_encoding_from_bytes(self, raw_bytes):
        """Detect encoding from raw bytes using charset-normalizer"""
        try:
            result = from_bytes(raw_bytes).best()
            if result is not None:
                encoding = result.encoding
                print(f"Detected encoding: {encoding}")
                return encoding
            else:
                print("No encoding detected. Using utf-8 as fallback.")
                return 'utf-8'
        except Exception as e:
            print(f"Encoding detection failed: {e}. Using utf-8 as fallback.")
            return 'utf-8'
    
    def download_raw_data(self, filename):
        """Download raw data from S3 with automatic encoding detection"""
        try:
            key = f"raw-data/{filename}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            raw_bytes = response['Body'].read()
            
            # Check if it's a CSV file
            if filename.lower().endswith('.csv'):
                print(f"Detecting encoding for CSV file: {filename}")
                
                # Detect encoding
                encoding = self.detect_encoding_from_bytes(raw_bytes)
                
                # Try to read as CSV with detected encoding
                try:
                    # Convert bytes to string with detected encoding
                    csv_content = raw_bytes.decode(encoding)
                    df = pd.read_csv(StringIO(csv_content))
                    print(f"Successfully loaded CSV with encoding: {encoding}")
                    return df
                    
                except UnicodeDecodeError:
                    print(f"Failed with {encoding}. Trying utf-8...")
                    try:
                        csv_content = raw_bytes.decode('utf-8')
                        df = pd.read_csv(StringIO(csv_content))
                        print("Successfully loaded CSV with utf-8 encoding")
                        return df
                    except UnicodeDecodeError:
                        print("Failed with utf-8. Trying latin-1...")
                        csv_content = raw_bytes.decode('latin-1')
                        df = pd.read_csv(StringIO(csv_content))
                        print("Successfully loaded CSV with latin-1 encoding")
                        return df
            
            else:
                # Handle JSON files (original behavior)
                data = json.loads(raw_bytes.decode('utf-8'))
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