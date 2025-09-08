import sys
from pathlib import Path

# Add config directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'config'))

from config import DATA_DIR
import pandas as pd
import random

# Create sample sentiment data
def create_sample_data(n_samples=1000):
    """Create sample sentiment analysis dataset"""
    
    # Sample positive and negative texts
    positive_texts = [
        "I love this product!",
        "Amazing quality and fast delivery",
        "Excellent customer service",
        "Highly recommend this item",
        "Perfect for my needs",
        "Outstanding value for money",
        "Great experience overall",
        "Will definitely buy again"
    ]
    
    negative_texts = [
        "Terrible product, waste of money",
        "Poor quality and slow delivery",
        "Worst customer service ever",
        "Would not recommend",
        "Complete disappointment",
        "Overpriced and low quality",
        "Regret this purchase",
        "Never buying from here again"
    ]
    
    # Generate random data
    data = []
    for i in range(n_samples):
        if random.random() < 0.5:
            text = random.choice(positive_texts)
            sentiment = 1
        else:
            text = random.choice(negative_texts)
            sentiment = 0
        
        data.append({
            'id': i,
            'text': text,
            'sentiment': sentiment,
            'created_at': pd.Timestamp.now()
        })
    
    return pd.DataFrame(data)

# Create and save sample data
if __name__ == "__main__":
    df = create_sample_data(1000)
    df.to_csv(DATA_DIR / 'sample_sentiment_data.csv', index=False)
    print(f"Created sample dataset with {len(df)} samples")
    print(f"Saved to: {DATA_DIR / 'sample_sentiment_data.csv'}")
    print("Sample data:")
    print(df.head())