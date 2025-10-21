"""
Configuration file for the MLOps Sentiment Analysis Pipeline
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
SRC_DIR = PROJECT_ROOT / 'src'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
TESTS_DIR = PROJECT_ROOT / 'tests'
CONFIG_DIR = PROJECT_ROOT / 'config'
MODELS_DIR = PROJECT_ROOT / 'models'

# AWS Configuration
# These will work with AWS CLI authentication (aws configure)
# boto3 will automatically pick up credentials from AWS CLI config
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'par-ty-sentiment-analysis-pipeline'  # Your S3 bucket name

# Optional: Override with environment variables if needed
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', AWS_REGION)
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', S3_BUCKET_NAME)

# Data Configuration
RAW_DATA_PATH = 'raw-data/'
PROCESSED_DATA_PATH = 'processed-data/'
MODELS_PATH = 'models/'
ARTIFACTS_PATH = 'artifacts/'

# Model Configuration
RANDOM_STATE = 42
TRAIN_SIZE = 0.7
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Environment detection
IS_LAMBDA = os.getenv('AWS_LAMBDA_FUNCTION_NAME') is not None
IS_LOCAL = not IS_LAMBDA

# Paths for different environments
if IS_LAMBDA:
    # Running in AWS Lambda
    TEMP_DIR = Path('/tmp')
    DATA_DIR = TEMP_DIR / 'data'
    MODELS_DIR = TEMP_DIR / 'models'
else:
    # Running locally
    TEMP_DIR = PROJECT_ROOT / 'temp'
    TEMP_DIR.mkdir(exist_ok=True)
