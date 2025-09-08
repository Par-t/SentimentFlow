import sys
from pathlib import Path

# Add directories to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'config'))

from data_ingestion import DataIngestion
from config import DATA_DIR
import pandas as pd

def test_data_pipeline():
    """Test the complete data pipeline"""
    
    # Initialize data ingestion (uses default bucket from config)
    ingestion = DataIngestion()
    
    # Load sample data using absolute path
    csv_path = DATA_DIR / 'sample_sentiment_data.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
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