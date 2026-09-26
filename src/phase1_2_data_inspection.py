import pandas as pd
import os
from pathlib import Path

def run_data_inspection():
    raw_data_path = Path("data/raw/Reviews.csv")
    processed_data_path = Path("data/processed/reviews_with_sentiment.csv")
    
    print("=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60)
    print(f"Loading raw data from: {raw_data_path.resolve()}")
    
    df = pd.read_csv(raw_data_path)
    
    print("\n" + "=" * 60)
    print("STEP 2: DATASET SHAPE")
    print("=" * 60)
    print(f"Total Rows (Reviews): {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    
    print("\n" + "=" * 60)
    print("STEP 3: FIRST 5 ROWS")
    print("=" * 60)
    print(df.head(5).to_string())
    
    print("\n" + "=" * 60)
    print("STEP 4: COLUMNS AND DATA TYPES")
    print("=" * 60)
    print(df.dtypes)
    
    print("\n" + "=" * 60)
    print("STEP 5: MISSING VALUES CHECK")
    print("=" * 60)
    missing_vals = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing_vals, 'Missing_Percentage': missing_pct})
    print(missing_df)
    
    print("\n" + "=" * 60)
    print("STEP 6: DUPLICATE REVIEWS CHECK")
    print("=" * 60)
    exact_duplicates = df.duplicated().sum()
    print(f"Exact row duplicates: {exact_duplicates:,}")
    
    # Text + UserId + Time duplicate check (same user reviewing same product/text at same time)
    text_user_time_dupes = df.duplicated(subset=['UserId', 'ProfileName', 'Time', 'Text']).sum()
    print(f"Review Content Duplicates (UserId + ProfileName + Time + Text): {text_user_time_dupes:,}")
    
    print("\n" + "=" * 60)
    print("STEP 7: SCORE DISTRIBUTION (1 to 5 Stars)")
    print("=" * 60)
    score_counts = df['Score'].value_counts().sort_index()
    score_pct = (df['Score'].value_counts(normalize=True).sort_index()) * 100
    score_df = pd.DataFrame({'Count': score_counts, 'Percentage (%)': score_pct.round(2)})
    print(score_df)
    
    print("\n" + "=" * 60)
    print("STEP 8: CREATING PROJECT-DEFINED SENTIMENT LABELS")
    print("=" * 60)
    def map_score_to_sentiment(score):
        if score <= 2:
            return 'Negative'
        elif score == 3:
            return 'Neutral'
        else:
            return 'Positive'
            
    df['Sentiment'] = df['Score'].apply(map_score_to_sentiment)
    print("Sentiment column created successfully from Score.")
    
    print("\n" + "=" * 60)
    print("STEP 9: SENTIMENT DISTRIBUTION")
    print("=" * 60)
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_pct = (df['Sentiment'].value_counts(normalize=True)) * 100
    sentiment_df = pd.DataFrame({'Count': sentiment_counts, 'Percentage (%)': sentiment_pct.round(2)})
    print(sentiment_df)
    
    print("\n" + "=" * 60)
    print("STEP 10: SAVING PROCESSED DATASET")
    print("=" * 60)
    # Ensure directory exists
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    print(f"Processed dataset saved with 'Sentiment' column to: {processed_data_path.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    run_data_inspection()
