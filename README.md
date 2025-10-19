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

**Option A: Using AWS CLI (Recommended)**
```bash
aws configure
```
Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format (e.g., `json`)

**Option B: Environment Variables**
Add AWS credentials to your virtual environment activation script:

**Windows (`venv\Scripts\activate.bat`):**
```batch
set AWS_ACCESS_KEY_ID=your_access_key_here
set AWS_SECRET_ACCESS_KEY=your_secret_key_here
set AWS_DEFAULT_REGION=us-east-1
```

**macOS/Linux (`venv/bin/activate`):**
```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
```

#### 9. Update Configuration

Edit `config/config.py` to use your S3 bucket:
```python
S3_BUCKET_NAME = 'your-bucket-name-here'  # Replace with your bucket name
AWS_REGION = 'your-region-here'           # Replace with your region
```

#### 10. Verify AWS Connection
```bash
aws s3 ls s3://your-bucket-name-here
```

### 🏗️ Project Structure

```
SentimentFlow/
├── src/                    # Source code
│   ├── data_ingestion.py   # S3 data upload/download
│   ├── data_loader.py      # Data loading and validation
│   ├── etl_pipeline.py     # Data preprocessing
│   ├── mlflow_utils.py     # MLflow integration
│   └── training_config.py  # Configuration management
├── config/                 # Configuration files
├── tests/                  # Test files
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

## 📝 Next Steps

- [ ] Data pipeline testing
- [ ] Model training workflows
- [ ] MLflow experiment tracking
- [ ] Model deployment guides

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here] 
