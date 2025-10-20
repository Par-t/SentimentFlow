"""
Quick test script to validate Steps 1, 2, and 3 setup
Run this to test configuration, MLflow setup, and data loading without experiments
"""
import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_step_1_dependencies():
    """Test Step 1: Dependencies and MLflow installation"""
    print("=" * 50)
    print("TESTING STEP 1: Dependencies & MLflow Installation")
    print("=" * 50)
    
    try:
        # Test MLflow import
        import mlflow
        print(f"✅ MLflow imported successfully - Version: {mlflow.__version__}")
        
        # Test other dependencies
        import pandas as pd
        print(f"✅ Pandas imported - Version: {pd.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn imported - Version: {sklearn.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib imported - Version: {matplotlib.__version__}")
        
        import seaborn as sns
        print(f"✅ Seaborn imported - Version: {sns.__version__}")
        
        # Test MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        print(f"✅ MLflow tracking URI set: {mlflow.get_tracking_uri()}")
        
        print("\n🎉 Step 1 PASSED: All dependencies working!")
        return True
        
    except Exception as e:
        print(f"❌ Step 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_step_2_training_config():
    """Test Step 2: Training Configuration Module"""
    print("\n" + "=" * 50)
    print("TESTING STEP 2: Training Configuration Module")
    print("=" * 50)
    
    try:
        # Test training config import
        from training_config import (
            TrainingConfig, 
            create_training_config,
            get_model_configs,
            get_data_config,
            get_mlflow_config,
            MODEL_CONFIGS,
            DATA_CONFIG,
            MLFLOW_CONFIG
        )
        print("✅ Training config module imported successfully")
        
        # Test creating default config
        config = create_training_config()
        print(f"✅ Default config created - S3 Bucket: {config.s3_bucket}")
        print(f"✅ Experiment Name: {config.experiment_name}")
        print(f"✅ Max Features: {config.max_features}")
        
        # Test model configs
        model_configs = get_model_configs()
        print(f"✅ Model configs loaded - {len(model_configs)} algorithms available:")
        for model_name in model_configs.keys():
            print(f"   - {model_name}")
        
        # Test data config
        data_config = get_data_config()
        print(f"✅ Data config loaded - Test size: {data_config['test_size']}")
        
        # Test MLflow config
        mlflow_config = get_mlflow_config()
        print(f"✅ MLflow config loaded - Tracking URI: {mlflow_config['tracking_uri']}")
        
        # Test creating custom config
        custom_config = create_training_config(
            experiment_name="test_experiment",
            max_features=1000
        )
        print(f"✅ Custom config created - Max features: {custom_config.max_features}")
        
        print("\n🎉 Step 2 PASSED: Training configuration working!")
        return True
        
    except Exception as e:
        print(f"❌ Step 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_step_3_mlflow_utils():
    """Test Step 3: MLflow Utilities Module"""
    print("\n" + "=" * 50)
    print("TESTING STEP 3: MLflow Utilities Module")
    print("=" * 50)
    
    try:
        # Test MLflow utils import
        from mlflow_utils import (
            setup_mlflow_experiment,
            create_experiment_if_not_exists,
            get_mlflow_client,
            log_experiment_info,
            start_mlflow_run,
            end_mlflow_run,
            compare_models,
            setup_mlflow_for_training
        )
        print("✅ MLflow utils module imported successfully")
        
        # Test MLflow client
        client = get_mlflow_client()
        print("✅ MLflow client created successfully")
        
        # Test experiment creation (without actually creating)
        print("✅ MLflow utilities functions available:")
        print("   - setup_mlflow_experiment()")
        print("   - create_experiment_if_not_exists()")
        print("   - log_experiment_info()")
        print("   - start_mlflow_run()")
        print("   - compare_models()")
        
        # Test configuration integration
        from training_config import MLFLOW_CONFIG
        print(f"✅ MLflow config integration working - URI: {MLFLOW_CONFIG['tracking_uri']}")
        
        print("\n🎉 Step 3 PASSED: MLflow utilities working!")
        return True
        
    except Exception as e:
        print(f"❌ Step 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_step_4_data_loader():
    """Test Step 4: Data Loading Module"""
    print("\n" + "=" * 50)
    print("TESTING STEP 4: Data Loading Module")
    print("=" * 50)
    
    try:
        # Test data loader import
        from src.data_loader import (
            DataLoader,
            load_training_data,
            split_data,
            prepare_features_labels,
            validate_data_quality
        )
        print("✅ Data loader module imported successfully")
        
        # Test DataLoader class
        loader = DataLoader()
        print(f"✅ DataLoader initialized - S3 Bucket: {loader.s3_bucket}")
        
        # Test configuration integration
        from training_config import DATA_CONFIG
        print(f"✅ Data config integration working - Test size: {DATA_CONFIG['test_size']}")
        
        # Test that all required functions are callable
        functions_to_test = [
            ('load_training_data', load_training_data),
            ('split_data', split_data),
            ('prepare_features_labels', prepare_features_labels),
            ('validate_data_quality', validate_data_quality),
        ]
        
        for func_name, func in functions_to_test:
            if callable(func):
                print(f"✅ {func_name} is callable")
            else:
                print(f"❌ {func_name} is not callable")
                return False
        
        print("\n🎉 Step 4 PASSED: Data loading module working!")
        return True
        
    except Exception as e:
        print(f"❌ Step 4 FAILED: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between all modules"""
    print("\n" + "=" * 50)
    print("TESTING INTEGRATION: All Modules Working Together")
    print("=" * 50)
    
    try:
        # Test imports
        from training_config import create_training_config, get_model_configs
        from mlflow_utils import get_mlflow_client, setup_mlflow_experiment
        from src.data_loader import DataLoader, load_training_data
        
        # Test configuration creation
        config = create_training_config(experiment_name="integration_test")
        print(f"✅ Config created with experiment: {config.experiment_name}")
        
        # Test model configs
        model_configs = get_model_configs()
        print(f"✅ {len(model_configs)} model configurations available")
        
        # Test MLflow client
        client = get_mlflow_client()
        print("✅ MLflow client working with config")
        
        # Test data loader
        loader = DataLoader()
        print("✅ DataLoader working with config")
        
        # Test that all required functions are callable
        functions_to_test = [
            ('create_training_config', create_training_config),
            ('get_model_configs', get_model_configs),
            ('get_mlflow_client', get_mlflow_client),
            ('load_training_data', load_training_data),
        ]
        
        for func_name, func in functions_to_test:
            if callable(func):
                print(f"✅ {func_name} is callable")
            else:
                print(f"❌ {func_name} is not callable")
                return False
        
        print("\n🎉 INTEGRATION PASSED: All modules working together!")
        return True
        
    except Exception as e:
        print(f"❌ INTEGRATION FAILED: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\n" + "=" * 50)
    print("TESTING FILE STRUCTURE")
    print("=" * 50)
    
    required_files = [
        'src/training_config.py',
        'src/mlflow_utils.py',
        'config/config.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    if all_exist:
        print("\n🎉 FILE STRUCTURE PASSED: All required files present!")
    else:
        print("\n❌ FILE STRUCTURE FAILED: Some files missing")
    
    return all_exist


def main():
    """Run all tests"""
    print("🚀 TESTING MLOPS SETUP - STEPS 1, 2, 3, 4")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Step 1: Dependencies", test_step_1_dependencies),
        ("Step 2: Training Config", test_step_2_training_config),
        ("Step 3: MLflow Utils", test_step_3_mlflow_utils),
        ("Step 4: Data Loader", test_step_4_data_loader),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for Hour 2!")
        print("\nNext: Create feature engineering and model training modules")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    main()
