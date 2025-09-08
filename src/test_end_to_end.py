import sys
from pathlib import Path

# Add directories to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'config'))

from data_ingestion import DataIngestion
from etl_pipeline import ETLPipeline
from config import DATA_DIR
import pandas as pd

def test_end_to_end_pipeline():
    """Test the complete data pipeline from raw to processed"""
    
    print("=== Testing End-to-End Data Pipeline ===")
    
    # Step 1: Data Ingestion
    print("\n1. Testing data ingestion...")
    ingestion = DataIngestion()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'text': [
            "I love this product! It's amazing.",
            "Terrible quality, waste of money.",
            "Good value for the price.",
            "Worst experience ever."
        ],
        'sentiment': [1, 0, 1, 0]
    })
    
    # Upload raw data
    success = ingestion.upload_raw_data(sample_data, "test_pipeline_data.json")
    if not success:
        print("❌ Data ingestion failed")
        return False
    
    print("✅ Data ingestion successful")
    
    # Step 2: ETL Processing
    print("\n2. Testing ETL pipeline...")
    etl = ETLPipeline()
    
    # Download and process data
    raw_data = ingestion.download_raw_data("test_pipeline_data.json")
    if raw_data is None:
        print("❌ Failed to download raw data")
        return False
    
    processed_data = etl.process_data(raw_data)
    etl.upload_processed_data(processed_data, "test_processed_data.json")
    
    print("✅ ETL pipeline successful")
    
    # Step 3: Verify results
    print("\n3. Verifying results...")
    print(f"Original data shape: {raw_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print("\nSample processed data:")
    print(processed_data[['text', 'cleaned_text', 'sentiment']].head())
    
    print("\n✅ End-to-end pipeline test completed successfully!")
    return True

if __name__ == "__main__":
    test_end_to_end_pipeline()
