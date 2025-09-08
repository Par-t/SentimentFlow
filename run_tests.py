#!/usr/bin/env python3
"""
Main test runner for the MLOps Sentiment Analysis Pipeline
"""
import sys
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / 'src'))

def main():
    """Run all pipeline tests"""
    print("🚀 Starting MLOps Sentiment Analysis Pipeline Tests")
    print("=" * 60)
    
    try:
        # Test 1: Create sample data
        print("\n📊 Test 1: Creating sample data...")
        from notebooks.create_sample_data import create_sample_data
        df = create_sample_data(100)
        print(f"✅ Created {len(df)} sample records")
        
        # Test 2: Data ingestion
        print("\n📤 Test 2: Testing data ingestion...")
        from src.test_data_pipeline import test_data_pipeline
        test_data_pipeline()
        
        # Test 3: ETL pipeline
        print("\n🔄 Test 3: Testing ETL pipeline...")
        from src.test_end_to_end import test_end_to_end_pipeline
        test_end_to_end_pipeline()
        
        print("\n🎉 All tests completed successfully!")
        print("Your MLOps pipeline is ready for the next phase!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
