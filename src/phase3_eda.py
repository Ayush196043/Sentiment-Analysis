import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_phase3_eda():
    input_path = Path("data/processed/reviews_with_sentiment.csv")
    dedup_path = Path("data/processed/reviews_deduplicated.csv")
    plots_dir = Path("notebooks/eda_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})
    
    print("=" * 70)
    print("PHASE 3 — DATA QUALITY CHECKS & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # STEP 1: VERIFY DATASET
    # ---------------------------------------------------------
    print("\n--- STEP 1: DATASET VERIFICATION ---")
    df = pd.read_csv(input_path)
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nData Types:\n", df.dtypes)
    print("\nFirst 5 Rows:\n", df[['Id', 'Score', 'Sentiment', 'Summary', 'Text']].head(5).to_string())
    
    # ---------------------------------------------------------
    # STEP 2: DUPLICATE ANALYSIS
    # ---------------------------------------------------------
    print("\n--- STEP 2: DUPLICATE ANALYSIS ---")
    total_rows = len(df)
    complete_dupes = df.duplicated().sum()
    text_dupes = df.duplicated(subset=['Text']).sum()
    composite_dupes = df.duplicated(subset=['UserId', 'ProfileName', 'Time', 'Text']).sum()
    
    print(f"1. Complete Duplicate Rows:  {complete_dupes:,} ({complete_dupes/total_rows*100:.2f}%)")
    print(f"2. Duplicate Review Texts:   {text_dupes:,} ({text_dupes/total_rows*100:.2f}%)")
    print(f"3. Composite Duplicates:     {composite_dupes:,} ({composite_dupes/total_rows*100:.2f}%)")
    
    df_dedup = df.drop_duplicates(subset=['Text']).copy()
    remaining_rows = len(df_dedup)
    print(f"Rows Remaining after dropping Duplicate Texts: {remaining_rows:,} ({remaining_rows/total_rows*100:.2f}% kept)")
    
    # Save deduplicated dataset separately
    df_dedup.to_csv(dedup_path, index=False)
    print(f"Saved deduplicated dataset to: {dedup_path.resolve()}")
    
    # ---------------------------------------------------------
    # STEP 3: MISSING VALUES
    # ---------------------------------------------------------
    print("\n--- STEP 3: MISSING VALUES ANALYSIS ---")
    missing = df_dedup[['Text', 'Score', 'Sentiment', 'Summary']].isnull().sum()
    missing_pct = (missing / len(df_dedup)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct.round(4)})
    print(missing_df)
    
    # Fill missing summary if any with empty string to avoid issues
    df_dedup['Summary'] = df_dedup['Summary'].fillna("")
    
    # ---------------------------------------------------------
    # STEP 4: SENTIMENT DISTRIBUTION
    # ---------------------------------------------------------
    print("\n--- STEP 4: SENTIMENT DISTRIBUTION (Deduplicated Dataset) ---")
    sentiment_counts = df_dedup['Sentiment'].value_counts()
    sentiment_pcts = df_dedup['Sentiment'].value_counts(normalize=True) * 100
    sent_df = pd.DataFrame({'Count': sentiment_counts, 'Percentage (%)': sentiment_pcts.round(2)})
    print(sent_df)
    
    # Plot Sentiment Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f1c40f'}
    bars = sns.barplot(x=sent_df.index, y=sent_df['Count'], hue=sent_df.index, palette=palette, legend=False, ax=ax)
    ax.set_title("Sentiment Distribution (Deduplicated Corpus)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Sentiment Category", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Reviews", fontsize=12, fontweight='bold')
    
    for p, pct in zip(ax.patches, sent_df['Percentage (%)']):
        height = p.get_height()
        ax.annotate(f"{height:,.0f}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    plt.tight_layout()
    fig.savefig(plots_dir / "02_sentiment_distribution.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # STEP 5: SCORE DISTRIBUTION
    # ---------------------------------------------------------
    print("\n--- STEP 5: SCORE DISTRIBUTION (Deduplicated Dataset) ---")
    score_counts = df_dedup['Score'].value_counts().sort_index()
    score_pcts = (df_dedup['Score'].value_counts(normalize=True).sort_index()) * 100
    score_df = pd.DataFrame({'Count': score_counts, 'Percentage (%)': score_pcts.round(2)})
    print(score_df)
    
    # Plot Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(x=score_df.index, y=score_df['Count'], color='#3498db', ax=ax)
    ax.set_title("Star Rating (Score) Distribution", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Star Rating (1 to 5 Stars)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Reviews", fontsize=12, fontweight='bold')
    
    for p, pct in zip(ax.patches, score_df['Percentage (%)']):
        height = p.get_height()
        ax.annotate(f"{height:,.0f}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height + 5000),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plots_dir / "01_score_distribution.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # STEP 6: REVIEW LENGTH ANALYSIS
    # ---------------------------------------------------------
    print("\n--- STEP 6: REVIEW LENGTH STATISTICS ---")
    df_dedup['text_char_count'] = df_dedup['Text'].astype(str).apply(len)
    df_dedup['text_word_count'] = df_dedup['Text'].astype(str).apply(lambda s: len(s.split()))
    
    char_stats = df_dedup['text_char_count'].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    word_stats = df_dedup['text_word_count'].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    
    stats_summary = pd.DataFrame({'Character Count': char_stats, 'Word Count': word_stats})
    print(stats_summary.round(2))
    
    # Plot Word Count Distribution (Truncated at 99th percentile for visualization clarity)
    p99_words = df_dedup['text_word_count'].quantile(0.99)
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [0.2, 0.8]})
    
    sns.boxplot(x=df_dedup[df_dedup['text_word_count'] <= p99_words]['text_word_count'], ax=ax_box, color='#9b59b6')
    ax_box.set(xlabel='')
    ax_box.set_title("Review Word Count Distribution (Up to 99th Percentile)", fontsize=14, fontweight='bold')
    
    sns.histplot(df_dedup[df_dedup['text_word_count'] <= p99_words]['text_word_count'], bins=50, kde=True, ax=ax_hist, color='#8e44ad')
    ax_hist.set_xlabel("Word Count per Review", fontsize=12, fontweight='bold')
    ax_hist.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax_hist.axvline(word_stats['mean'], color='red', linestyle='--', label=f"Mean: {word_stats['mean']:.1f}")
    ax_hist.axvline(word_stats['50%'], color='green', linestyle='-', label=f"Median: {word_stats['50%']:.1f}")
    ax_hist.legend(fontsize=11)
    
    plt.tight_layout()
    fig.savefig(plots_dir / "03_review_word_count_distribution.png", dpi=300)
    plt.close()
    
    # Plot Character Count Distribution
    p99_chars = df_dedup['text_char_count'].quantile(0.99)
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [0.2, 0.8]})
    
    sns.boxplot(x=df_dedup[df_dedup['text_char_count'] <= p99_chars]['text_char_count'], ax=ax_box, color='#16a085')
    ax_box.set(xlabel='')
    ax_box.set_title("Review Character Count Distribution (Up to 99th Percentile)", fontsize=14, fontweight='bold')
    
    sns.histplot(df_dedup[df_dedup['text_char_count'] <= p99_chars]['text_char_count'], bins=50, kde=True, ax=ax_hist, color='#1be3b9')
    ax_hist.set_xlabel("Character Count per Review", fontsize=12, fontweight='bold')
    ax_hist.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax_hist.axvline(char_stats['mean'], color='red', linestyle='--', label=f"Mean: {char_stats['mean']:.1f}")
    ax_hist.axvline(char_stats['50%'], color='green', linestyle='-', label=f"Median: {char_stats['50%']:.1f}")
    ax_hist.legend(fontsize=11)
    
    plt.tight_layout()
    fig.savefig(plots_dir / "04_review_char_count_distribution.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # STEP 7: EXTREME / UNUSUAL REVIEWS
    # ---------------------------------------------------------
    print("\n--- STEP 7: EXTREME REVIEWS INSPECTION ---")
    very_short = df_dedup[df_dedup['text_word_count'] <= 3]
    very_long = df_dedup[df_dedup['text_word_count'] >= 500]
    
    print(f"Reviews with <= 3 words: {len(very_short):,} ({len(very_short)/len(df_dedup)*100:.2f}%)")
    print(f"Reviews with >= 500 words: {len(very_long):,} ({len(very_long)/len(df_dedup)*100:.2f}%)")
    
    print("\nSample Extremely Short Reviews (<= 3 words):")
    for idx, row in very_short[['Score', 'Sentiment', 'Text']].head(5).iterrows():
        print(f"  - [{row['Sentiment']}] Score {row['Score']}: \"{row['Text']}\"")
        
    print("\nSample Extremely Long Reviews (>= 500 words snippet):")
    for idx, row in very_long[['Score', 'Sentiment', 'Text', 'text_word_count']].head(3).iterrows():
        print(f"  - [{row['Sentiment']}] Score {row['Score']} ({row['text_word_count']} words): \"{row['Text'][:150]}...\"")
        
    # ---------------------------------------------------------
    # STEP 8: SCORE VS SENTIMENT RELATIONSHIP CROSS-TAB
    # ---------------------------------------------------------
    print("\n--- STEP 8: SCORE vs SENTIMENT CROSS-TABULATION ---")
    ct = pd.crosstab(df_dedup['Score'], df_dedup['Sentiment'], margins=True)
    print(ct)
    
    # Plot Cross-Tab
    fig, ax = plt.subplots(figsize=(8, 5))
    ct_no_margin = pd.crosstab(df_dedup['Score'], df_dedup['Sentiment'])[['Negative', 'Neutral', 'Positive']]
    ct_no_margin.plot(kind='bar', stacked=True, color=['#e74c3c', '#f1c40f', '#2ecc71'], ax=ax)
    ax.set_title("Score to Sentiment Mapping Verification", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Star Rating (Score)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Reviews", fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.legend(title="Sentiment Category", fontsize=11)
    plt.tight_layout()
    fig.savefig(plots_dir / "05_score_vs_sentiment_cross_check.png", dpi=300)
    plt.close()
    
    print("\n=" * 70)
    print("PHASE 3 EDA COMPLETED SUCCESSFULLY!")
    print(f"Plots saved to: {plots_dir.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    run_phase3_eda()
