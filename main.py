#!/usr/bin/env python3
"""
SentimentFlow Pipeline - Main Entry Point

A minimalist CLI interface for running the complete sentiment analysis pipeline
on real datasets stored in S3.

Usage:
    python main.py --dataset mydataset.csv --label-column sentiment --query-column text
"""

import argparse
import sys
from pathlib import Path

# Add project directories to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'config'))

from data_ingestion import DataIngestion
from etl_pipeline import ETLPipeline
from feature_extraction import feature_extraction
from simple_tracker import SimpleTracker
from config import S3_BUCKET_NAME
import pandas as pd


def main():
    """Main entry point for the SentimentFlow pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run SentimentFlow pipeline on S3 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset sentimentdataset.csv --label-column sentiment --query-column text
  python main.py --dataset reviews.csv --label-column rating --query-column review_text
        """
    )
    
    parser.add_argument(
        '--dataset', 
        required=True,
        help='Dataset filename in S3 (e.g., mydataset.csv)'
    )
    
    parser.add_argument(
        '--label-column',
        required=True,
        help='Name of the column containing sentiment labels'
    )
    
    parser.add_argument(
        '--query-column',
        required=True,
        help='Name of the column containing text data to analyze'
    )
    
    parser.add_argument(
        '--bucket',
        default=S3_BUCKET_NAME,
        help=f'S3 bucket name (default: {S3_BUCKET_NAME})'
    )
    
    parser.add_argument(
        '--output-prefix',
        default='pipeline_output',
        help='Prefix for output files in S3 (default: pipeline_output)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("━━ SentimentFlow Pipeline ━━")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Label Column: {args.label_column}")
    print(f"Query Column: {args.query_column}")
    print(f"S3 Bucket: {args.bucket}")
    print("=" * 60)
    
    try:
        # Initialize pipeline components
        print("\n🔧 Initializing pipeline components...")
        ingestion = DataIngestion(bucket_name=args.bucket)
        etl = ETLPipeline(bucket_name=args.bucket)
        tracker = SimpleTracker(bucket_name=args.bucket)
        
        # Step 1: Download dataset from S3
        print(f"\n📥 Step 1: Downloading dataset from S3...")
        print(f"   Downloading: {args.dataset}")
        
        raw_data = ingestion.download_raw_data(args.dataset)
        if raw_data is None:
            print(f"❌ Failed to download dataset: {args.dataset}")
            return 1
        
        print(f"✅ Downloaded dataset with {len(raw_data)} rows and {len(raw_data.columns)} columns")
        print(f"   Columns: {list(raw_data.columns)}")
        
        # Step 2: Validate required columns exist
        print(f"\n🔍 Step 2: Validating dataset structure...")
        
        if args.label_column not in raw_data.columns:
            print(f"❌ Label column '{args.label_column}' not found in dataset")
            print(f"   Available columns: {list(raw_data.columns)}")
            return 1
        
        if args.query_column not in raw_data.columns:
            print(f"❌ Query column '{args.query_column}' not found in dataset")
            print(f"   Available columns: {list(raw_data.columns)}")
            return 1
        
        print(f"✅ Required columns found")
        
        # Step 3: Process data with ETL pipeline
        print(f"\n⚙️  Step 3: Processing data with ETL pipeline...")
        print(f"   Applying cleaning functions and label mapping...")
        
        # Rename columns to standard names for pipeline compatibility
        data_copy = raw_data.copy()
        data_copy['text'] = data_copy[args.query_column]
        data_copy['sentiment'] = data_copy[args.label_column]
        
        # Process data through ETL pipeline
        processed_data = etl.process_data(data_copy, tracker=tracker, dataset_name=args.dataset)
        
        if processed_data is None or len(processed_data) == 0:
            print("❌ ETL processing failed or resulted in empty dataset")
            return 1
        
        print(f"✅ ETL processing completed")
        print(f"   Processed {len(processed_data)} rows")
        
        # Step 4: Feature extraction
        print(f"\n🧠 Step 4: Extracting features...")
        
        feature_matrix, feature_names, extractor = etl.feature_extraction(processed_data, tracker=tracker, dataset_name=args.dataset)
        
        if feature_matrix is None:
            print("❌ Feature extraction failed")
            return 1
        
        print(f"✅ Feature extraction completed")
        print(f"   Feature matrix shape: {feature_matrix.shape}")
        print(f"   Number of features: {len(feature_names)}")
        
        # Step 5: Upload processed data and artifacts
        print(f"\n📤 Step 5: Uploading results to S3...")
        
        # Upload processed data
        processed_filename = f"{args.output_prefix}_processed_{args.dataset}"
        upload_success = etl.upload_processed_data(processed_data, processed_filename)
        
        if not upload_success:
            print("❌ Failed to upload processed data")
            return 1
        
        # Track the processed data file
        tracker.add_artifact(args.dataset, "processed_data_file", processed_filename)
        
        print(f"✅ Results uploaded successfully")
        print(f"   Processed data: {processed_filename}")
        
        # Step 6: Summary and next steps
        print(f"\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        print("📊 Pipeline Summary:")
        print(f"   • Original dataset: {len(raw_data)} rows")
        print(f"   • Processed dataset: {len(processed_data)} rows")
        print(f"   • Features extracted: {len(feature_names)}")
        print(f"   • Feature matrix shape: {feature_matrix.shape}")
        print("=" * 60)
        print("\n💡 Next Steps:")
        print("   • Use the processed data for model training")
        print("   • Feature matrix is ready for ML algorithms")
        print("   • All artifacts saved to S3 for reproducibility")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
