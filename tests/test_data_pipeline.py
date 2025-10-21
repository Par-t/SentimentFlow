import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

# Add directories to Python path for i
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'config'))
sys.path.append(str(PROJECT_ROOT / 'notebooks'))

from data_ingestion import DataIngestion
from config import DATA_DIR
import pandas as pd

def test_data_pipeline():
    """Test the complete data pipeline"""
    
    # Initialize data ingestion (uses default bucket from config)
    ingestion = DataIngestion()
    
    # Create sample data instead of loading from file
    from create_sample_data import create_sample_data
    df = create_sample_data(50)  # Create 50 sample records
    print(f"Created {len(df)} sample records")
    
    # Upload to S3
    success = ingestion.upload_raw_data(df, "sample_sentiment_data.json")
    
    if success:
        # Download and verify
        downloaded_df = ingestion.download_raw_data("sample_sentiment_data.json")
        print(f"Downloaded {len(downloaded_df)} samples")
        print("Data verification:")
        print(f"Original shape: {df.shape}")
        print(f"Downloaded shape: {downloaded_df.shape}")
        print("First few rows:")
        print(downloaded_df.head())
    else:
        print("Data upload failed")

if __name__ == "__main__":
    test_data_pipeline()