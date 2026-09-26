import pandas as pd
from pathlib import Path

def verify_phase2():
    raw_path = Path("data/raw/Reviews.csv")
    proc_path = Path("data/processed/reviews_with_sentiment.csv")
    
    df_raw = pd.read_csv(raw_path)
    df_proc = pd.read_csv(proc_path)
    
    print("=" * 65)
    print("PHASE 2 VERIFICATION REPORT")
    print("=" * 65)
    
    # 1. Row counts
    print(f"1. Original dataset row count:  {len(df_raw):,}")
    print(f"2. Processed dataset row count: {len(df_proc):,}")
    
    # 2. Duplicate checks
    print("\n--- DUPLICATE CHECKS BREAKDOWN ---")
    complete_row_dupes = df_raw.duplicated().sum()
    print(f"A. Complete duplicate rows (all columns identical): {complete_row_dupes:,}")
    
    text_only_dupes = df_raw.duplicated(subset=['Text']).sum()
    print(f"B. Duplicate review texts ('Text' column alone):    {text_only_dupes:,}")
    
    composite_dupes = df_raw.duplicated(subset=['UserId', 'ProfileName', 'Time', 'Text']).sum()
    print(f"C. Composite duplicates (UserId + ProfileName + Time + Text): {composite_dupes:,}")
    
    rows_remaining_unique_text = df_raw.drop_duplicates(subset=['Text']).shape[0]
    print(f"D. Rows remaining after removing duplicate review texts: {rows_remaining_unique_text:,}")
    
    # 3. Sentiment Distribution
    print("\n--- SENTIMENT LABELS BREAKDOWN ---")
    sentiment_counts = df_proc['Sentiment'].value_counts()
    sentiment_pcts = df_proc['Sentiment'].value_counts(normalize=True) * 100
    
    summary_df = pd.DataFrame({
        'Count': sentiment_counts,
        'Percentage (%)': sentiment_pcts.round(2)
    })
    print(summary_df)
    
    # 4. Code verification
    print("\n--- EXACT CODE USED FOR SENTIMENT MAPPING ---")
    print("""
def map_score_to_sentiment(score):
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['Sentiment'] = df['Score'].apply(map_score_to_sentiment)
    """)
    print("=" * 65)

if __name__ == "__main__":
    verify_phase2()
