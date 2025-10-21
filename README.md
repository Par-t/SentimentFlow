# SentimentFlow

A comprehensive MLOps pipeline for sentiment analysis with AWS S3 integration, MLflow tracking, and automated model training.

## 🚀 Quick Start

### Prerequisites

- Python 3.11.9 or higher
- AWS Account with S3 access
- Git

### Installation & Setup

#### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd SentimentFlow
```

#### 2. Install Python 3.11.9
If you don't have Python 3.11.9 installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python@3.11`
- **Linux**: `sudo apt install python3.11` or use your package manager

#### 3. Create Virtual Environment
```bash
python -m venv venv
```

#### 4. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

#### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 6. Install AWS CLI
Download and install AWS CLI from [aws.amazon.com/cli](https://aws.amazon.com/cli/)

**Windows:**
- Download the MSI installer and run it
- Or use: `winget install Amazon.AWSCLI`

**macOS:**
```bash
brew install awscli
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

#### 7. Set Up AWS S3 Bucket

**Create an S3 Bucket:**
1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com/)
2. Click "Create bucket"
3. Choose a unique bucket name (e.g., `your-name-sentiment-analysis-pipeline`)
4. Select your preferred region (e.g., `us-east-1`)
5. Leave other settings as default and create the bucket

**Set Up IAM User with S3 Permissions:**
1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Create a new user or use an existing one
3. Attach the `AmazonS3FullAccess` policy (or create a custom policy with S3 permissions)
4. Create access keys for the user

#### 8. Configure AWS Credentials

**Using AWS CLI (Recommended)**
```bash
aws configure
```
Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format (e.g., `json`)

**Verify AWS CLI Setup:**
```bash
aws s3 ls
```

**Test AWS Authentication:**
```bash
python test_aws_cli_auth.py
```

#### 9. Update Configuration (Optional)

If you want to use a different bucket name, edit `config/config.py`:
```python
S3_BUCKET_NAME = 'your-bucket-name-here'  # Replace with your bucket name
AWS_REGION = 'your-region-here'           # Replace with your region
```

#### 10. Verify AWS Connection
```bash
aws s3 ls s3://your-bucket-name-here
```

---

## 🚀 Running the Pipeline

### 1. Upload Dataset to S3
```bash
aws s3 cp {local_dataset_path} s3://{bucket_name}/raw-data/{dataset_filename}
```

### 2. Run Pipeline
```bash
python main.py --dataset {dataset_filename} --label-column {label_column_name} --query-column {text_column_name}
```

**Example:**
```bash
aws s3 cp data/sentimentdataset.csv s3://my-bucket/raw-data/sentimentdataset.csv
python main.py --dataset sentimentdataset.csv --label-column sentiment --query-column text
```

**Optional parameters:**
- `--bucket` - S3 bucket name
- `--output-prefix` - Output file prefix

---

## 🧪 Testing the Pipeline

Once you've completed the setup, you can test the **data preprocessing pipeline** to ensure everything is working correctly.

### What the Pipeline Currently Does

The current implementation focuses on the **data foundation** of the MLOps pipeline:

1. **📤 Data Ingestion** - Upload/download data to/from S3
2. **🔄 ETL Processing** - Clean and preprocess text data using NLP
3. **💾 Data Storage** - Organize data in S3 bucket structure
4. **✅ Validation** - Verify end-to-end data flow works correctly

### Run the Complete Test Suite
```bash
python run_tests.py
```

This will run a comprehensive test that includes:
1. **Sample Data Creation** - Creates test sentiment data (100 records)
2. **Data Ingestion** - Tests S3 upload/download functionality  
3. **ETL Pipeline** - Tests text cleaning and preprocessing
4. **End-to-End Flow** - Tests complete pipeline from raw data to processed data

### Run Individual Tests

**Test Complete End-to-End Pipeline:**
```bash
python tests/test_end_to_end.py
```
Tests: Raw data → S3 upload → Download → Text cleaning → Feature extraction → S3 upload → Verification

**Test Data Ingestion Only:**
```bash
python tests/test_data_pipeline.py
```
Tests: Sample data creation → S3 upload → Download → Verification

### Expected Test Output
If everything is working correctly, you should see:
- ✅ Sample data creation successful
- ✅ Data ingestion to S3 successful  
- ✅ ETL processing successful (text cleaning, stopword removal, lemmatization)
- ✅ Feature extraction successful (TF-IDF vectorization, feature selection)
- ✅ End-to-end pipeline test completed
- 📊 Data shape verification (original vs processed vs feature matrix)

### Sample Test Output
```
=== Testing End-to-End Data Pipeline ===

1. Testing data ingestion...
✅ Data ingestion successful

2. Testing ETL pipeline...
✅ ETL pipeline successful

3. Testing feature extraction...
✅ Feature extraction successful

4. Verifying results...
Original data shape: (4, 2)
Processed data shape: (4, 5)
Feature matrix shape: (4, 15)
Number of features: 15
Sample processed data:
                           text                    cleaned_text  sentiment
0  I love this product! It's amazing.  love product amazing          1
1  Terrible quality, waste of money.  terrible quality waste money   0

Sample features: ['amazing', 'bad', 'experience', 'good', 'love', ...]

✅ End-to-end pipeline test completed successfully!
```

### What Gets Created in S3
After running tests, your S3 bucket will contain:
```
s3://your-bucket/
├── raw-data/
│   ├── test_pipeline_data.json          # Raw test data
│   └── sample_sentiment_data.json       # Sample data from ingestion test
├── processed-data/
│   ├── test_processed_data.json         # Cleaned and processed data
│   └── processed_sample_sentiment_data.json
└── artifacts/
    └── feature_extractor_YYYYMMDD_HHMMSS.pkl  # Fitted TF-IDF vectorizer
```

### Troubleshooting Tests
- **AWS Connection Issues**: Verify your credentials and bucket permissions
- **Import Errors**: Make sure your virtual environment is activated
- **S3 Permission Errors**: Ensure your AWS user has S3 read/write permissions
- **Empty Results**: Check that your S3 bucket name in `config/config.py` is correct

### 🏗️ Project Structure

```
SentimentFlow/
├── src/                    # Source code
│   ├── data_ingestion.py   # S3 data upload/download
│   ├── data_loader.py      # Data loading and validation
│   ├── etl_pipeline.py     # Data preprocessing and feature extraction
│   ├── feature_extraction.py # TF-IDF vectorization and feature selection
│   ├── mlflow_utils.py     # MLflow integration
│   └── training_config.py  # Configuration management
├── tests/                  # Test files
│   ├── test_end_to_end.py  # Complete pipeline test
│   └── test_data_pipeline.py # Data ingestion test
├── config/                 # Configuration files
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### 📊 S3 Bucket Structure

The pipeline expects the following S3 bucket structure:
```
s3://your-bucket-name/
├── raw-data/              # Raw input data
├── processed-data/        # Cleaned and processed data
├── models/                # Trained model artifacts
└── artifacts/             # Additional artifacts (plots, logs, etc.)
```

**Note:** The folders will be created automatically when you first run the pipeline.

### 🔧 Configuration

Key configuration is managed in `config/config.py`:
- **S3 Bucket**: Update with your bucket name
- **AWS Region**: Update with your preferred region
- **Data Paths**: Configured for the expected S3 structure

**Important:** Make sure to update the configuration file with your own AWS details before running the pipeline.

### 💰 AWS Costs & Considerations

**S3 Storage Costs:**
- S3 charges for storage, requests, and data transfer
- For development/testing, costs are typically minimal (< $1/month)
- Monitor your usage in the AWS Billing Dashboard

**IAM Best Practices:**
- Use least-privilege access (only S3 permissions needed)
- Consider using IAM roles instead of access keys when possible
- Rotate access keys regularly for security

**Free Tier:**
- New AWS accounts get 12 months of free tier benefits
- Includes 5GB of S3 storage and limited requests

---

## 📝 Current Status & Next Steps

### ✅ **Completed (Data Foundation)**
- [x] **Setup and Installation** - Complete
- [x] **AWS S3 Integration** - Complete
- [x] **Data Ingestion Pipeline** - Complete (upload/download to S3)
- [x] **ETL Pipeline** - Complete (text cleaning, preprocessing)
- [x] **Data Validation** - Complete (end-to-end testing)
- [x] **Testing Framework** - Complete

### 🔄 **Next Phase (Model Training)**
- [ ] **Feature Engineering** - TF-IDF vectorization, feature selection
- [ ] **Model Training** - Implement sentiment classification models
- [ ] **Model Evaluation** - Cross-validation, metrics, comparison
- [ ] **MLflow Integration** - Experiment tracking and model versioning

### 🚀 **Future Phase (Production)**
- [ ] **Model Deployment** - API endpoints, inference pipeline
- [ ] **Monitoring** - Model performance tracking
- [ ] **CI/CD Pipeline** - Automated training and deployment

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here] 
