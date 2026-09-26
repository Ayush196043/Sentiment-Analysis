import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

from aspect_analysis import ASPECT_KEYWORDS, analyze_aspect_sentiment, analyze_clause_sentiment

def run_phase10():
    data_path = Path("data/processed/reviews_cleaned.csv")
    results_dir = Path("results")
    plots_dir = Path("notebooks/evaluation_plots")
    nb_dir = Path("notebooks")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot formatting
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})
    
    print("=" * 75)
    print("PHASE 10 — ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
    print("=" * 75)
    
    # ---------------------------------------------------------
    # STEP 1 & 2: CONFIRM ASPECT DICTIONARIES
    # ---------------------------------------------------------
    print("\n--- STEP 1 & 2: ASPECT CATEGORIES & KEYWORD DICTIONARIES ---")
    for aspect, kws in ASPECT_KEYWORDS.items():
        print(f"  • {aspect:<18}: {len(kws)} keywords -> Sample: {kws[:6]}")
        
    # ---------------------------------------------------------
    # STEP 7: DEMONSTRATING MIXED SENTIMENT ABSA EXAMPLES
    # ---------------------------------------------------------
    print("\n--- STEP 7: DEMONSTRATING MIXED SENTIMENT EXAMPLES ---")
    test_reviews = [
        "The taste is amazing and delicious, but the delivery was late and packaging was damaged.",
        "Good quality coffee, but overpriced and expensive.",
        "Not fresh at all, stale taste, but customer service gave a full refund fast.",
        "The ingredients are organic and healthy, but it tastes bland and terrible."
    ]
    
    for i, review in enumerate(test_reviews, 1):
        print(f"\n[Test Review {i}]: \"{review}\"")
        absa_results = analyze_aspect_sentiment(review)
        for item in absa_results:
            print(f"   -> Aspect: {item['aspect']:<18} | Sentiment: {item['sentiment']:<8} | Clause: \"{item['matching_clause']}\"")
            
    # ---------------------------------------------------------
    # STEP 8 & 9: DATASET-LEVEL ABSA ANALYSIS (78,716 TEST REVIEWS)
    # ---------------------------------------------------------
    print("\n--- STEP 8 & 9: RUNNING ABSA PIPELINE ACROSS TEST SET (78,716 REVIEWS) ---")
    df = pd.read_csv(data_path)
    df['clean_text'] = df['clean_text'].fillna("")
    
    _, X_test_raw, _, y_test = train_test_split(
        df['clean_text'], df['Sentiment'], test_size=0.20, random_state=42, stratify=df['Sentiment']
    )
    
    df_test = df.iloc[X_test_raw.index].copy()
    total_test = len(df_test)
    print(f"Analyzing {total_test:,} test reviews...")
    
    all_absa_records = []
    
    for idx, row in df_test.iterrows():
        text = str(row['Text'])
        extracted = analyze_aspect_sentiment(text)
        for item in extracted:
            all_absa_records.append({
                'Review_Id': row['Id'],
                'Overall_Score': row['Score'],
                'Overall_Sentiment': row['Sentiment'],
                'Aspect': item['aspect'],
                'Aspect_Sentiment': item['sentiment'],
                'Clause_Snippet': item['matching_clause']
            })
            
    absa_df = pd.DataFrame(all_absa_records)
    print(f"Total Aspect Mentions Extracted: {len(absa_df):,} mentions across {total_test:,} reviews.")
    
    # Save detailed ABSA results CSV
    absa_csv_path = results_dir / "aspect_sentiment_results.csv"
    absa_df.to_csv(absa_csv_path, index=False)
    print(f"Saved detailed ABSA extractions to: {absa_csv_path.resolve()}")
    
    # ---------------------------------------------------------
    # STEP 9 & 10: ASPECT SUMMARY & COMMON COMPLAINTS TABLE
    # ---------------------------------------------------------
    print("\n--- STEP 9 & 10: ASPECT SUMMARY & COMMON COMPLAINTS ---")
    
    aspect_stats = []
    for aspect in ASPECT_KEYWORDS.keys():
        sub = absa_df[absa_df['Aspect'] == aspect]
        total_mentions = len(sub)
        pct_of_reviews = (total_mentions / total_test) * 100
        
        pos_cnt = (sub['Aspect_Sentiment'] == 'Positive').sum()
        neg_cnt = (sub['Aspect_Sentiment'] == 'Negative').sum()
        neu_cnt = (sub['Aspect_Sentiment'] == 'Neutral').sum()
        
        neg_ratio = (neg_cnt / total_mentions * 100) if total_mentions > 0 else 0
        
        aspect_stats.append({
            'Aspect': aspect,
            'Total Mentions': total_mentions,
            'Mention % of Reviews': round(pct_of_reviews, 2),
            'Positive Mentions': pos_cnt,
            'Negative Mentions': neg_cnt,
            'Neutral Mentions': neu_cnt,
            'Negative Mention Ratio (%)': round(neg_ratio, 2)
        })
        
    summary_df = pd.DataFrame(aspect_stats).sort_values(by='Total Mentions', ascending=False)
    print(summary_df.to_string(index=False))
    
    summary_csv_path = results_dir / "aspect_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved aspect summary table to: {summary_csv_path.resolve()}")
    
    # ---------------------------------------------------------
    # STEP 9: VISUALIZATIONS FOR ABSA
    # ---------------------------------------------------------
    print("\n--- GENERATING ABSA VISUALIZATION CHARTS ---")
    
    # Chart 1: Aspect Mention Frequencies
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = sns.barplot(x=summary_df['Total Mentions'], y=summary_df['Aspect'], color='#3498db', ax=ax)
    ax.set_title("Customer Feedback Aspect Mention Frequency", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Total Mention Count", fontsize=12, fontweight='bold')
    ax.set_ylabel("Aspect Category", fontsize=12, fontweight='bold')
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:,.0f}", (width + 500, p.get_y() + p.get_height()/2.), ha='left', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plots_dir / "05_aspect_mention_frequencies.png", dpi=300)
    plt.close()
    
    # Chart 2: Aspect Sentiment Breakdown (Stacked Bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_plot_df = summary_df.set_index('Aspect')[['Positive Mentions', 'Neutral Mentions', 'Negative Mentions']]
    summary_plot_df.plot(kind='barh', stacked=True, color=['#2ecc71', '#f1c40f', '#e74c3c'], ax=ax)
    ax.set_title("Aspect-Wise Sentiment Distribution (Positive vs Neutral vs Negative)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Mentions", fontsize=12, fontweight='bold')
    ax.set_ylabel("Aspect Category", fontsize=12, fontweight='bold')
    plt.legend(title="Aspect Sentiment", fontsize=11)
    plt.tight_layout()
    fig.savefig(plots_dir / "06_aspect_sentiment_breakdown.png", dpi=300)
    plt.close()
    
    print(f"Saved ABSA charts to: {plots_dir.resolve()}")

    # ---------------------------------------------------------
    # STEP 11 & 12: CREATE NOTEBOOK notebooks/08_aspect_based_sentiment.ipynb
    # ---------------------------------------------------------
    print("\n--- CREATING NOTEBOOK notebooks/08_aspect_based_sentiment.ipynb ---")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 10 — Aspect-Based Sentiment Analysis (ABSA) Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Extract domain-specific aspect categories (Taste, Packaging, Delivery, Price, Quality, Ingredients, Support) and assign sentence-level local sentiment.\n",
                    "\n",
                    "### ABSA Workflow:\n",
                    "1. Define domain aspect keyword dictionaries.\n",
                    "2. Split customer reviews into sentence clauses.\n",
                    "3. Locate aspect keywords and analyze local clause sentiment (handling negations).\n",
                    "4. Run ABSA across 78,716 test reviews.\n",
                    "5. Compute aspect mention frequencies and common complaint rankings."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import sys\n",
                    "sys.path.append('../src')\n",
                    "from aspect_analysis import ASPECT_KEYWORDS, analyze_aspect_sentiment\n",
                    "\n",
                    "review = 'The taste is excellent and delicious, but the delivery was late and packaging was damaged.'\n",
                    "results = analyze_aspect_sentiment(review)\n",
                    "display(pd.DataFrame(results))"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    nb_path = nb_dir / "08_aspect_based_sentiment.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {nb_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase10()
