# SentimentFlow

An MLOps pipeline for sentiment analysis with AWS S3 integration, data preprocessing, and feature extraction.

## Prerequisites

- Python 3.11.9 or higher
- AWS Account with S3 access
- AWS CLI installed and configured

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd SentimentFlow
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
```bash
aws configure
```

5. Create an S3 bucket and update `config/config.py` with your bucket name:
```python
S3_BUCKET_NAME = 'your-bucket-name-here'
```

## Usage

1. Upload your dataset to S3:
```bash
aws s3 cp data/sentimentdataset.csv s3://par-ty-sentiment-analysis-pipeline/raw-data/sentimentdataset.csv
```

2. Run the pipeline:
```bash
python main.py --dataset sentimentdataset.csv --label-column sentiment --query-column text
```

## What the Pipeline Does

1. **Data Processing**: Cleans and preprocesses text data (lowercase, remove special characters, tokenization, lemmatization)
2. **Label Mapping**: Automatically maps text labels to numeric values
3. **Data Splitting**: Splits data into train/test/validation sets (70/20/10)
4. **Feature Extraction**: Extracts TF-IDF features, fitting only on training data to prevent data leakage
5. **Artifact Storage**: Saves all processed data, splits, features, and parameters to S3

## Output Structure

The pipeline creates the following S3 structure:
```
s3://your-bucket/
├── processed_data/           # Cleaned data
├── data_splits/
│   ├── train/               # Training splits
│   ├── test/                # Test splits
│   └── val/                 # Validation splits
├── features/
│   ├── train/               # Training features
│   ├── test/                # Test features
│   └── val/                 # Validation features
└── artifacts/               # Feature extractors, label mappings, parameters
```

## Project Structure

```
SentimentFlow/
├── src/                    # Source code
│   ├── data_ingestion.py   # S3 data upload/download
│   ├── etl_pipeline.py     # Data preprocessing and splitting
│   ├── feature_extraction.py # Feature extraction
│   └── simple_tracker.py   # Experiment tracking
├── config/                 # Configuration files
├── data/                   # Local data files
└── main.py                # Pipeline entry point
```

## Configuration

Key settings in `config/config.py`:
- `S3_BUCKET_NAME`: Your S3 bucket name
- `TRAIN_SIZE`, `TEST_SIZE`, `VAL_SIZE`: Data split ratios
- `RANDOM_STATE`: Random seed for reproducible splits