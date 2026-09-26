import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def run_phase5():
    input_path = Path("data/processed/reviews_cleaned.csv")
    nb_path = Path("notebooks/03_train_test_split.ipynb")
    
    print("=" * 75)
    print("PHASE 5 — TRAIN / TEST SPLIT & STRATIFICATION")
    print("=" * 75)
    
    # 1. Load Data
    print("\n--- STEP 1: LOADING CLEANED DATASET ---")
    df = pd.read_csv(input_path)
    print(f"Dataset Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Missing clean_text values: {df['clean_text'].isnull().sum()}")
    print(f"Missing Sentiment values : {df['Sentiment'].isnull().sum()}")
    
    # Fill any subtle null string if present
    df['clean_text'] = df['clean_text'].fillna("")
    
    X = df['clean_text']
    y = df['Sentiment']
    
    # 2. Train / Test Split
    print("\n--- STEP 2: PERFORMING TRAIN/TEST SPLIT (80/20, random_state=42, stratify=y) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    total_len = len(df)
    train_len = len(X_train)
    test_len = len(X_test)
    
    print(f"Total Rows   : {total_len:,} (100.00%)")
    print(f"Training Rows: {train_len:,} ({train_len/total_len*100:.2f}%)")
    print(f"Testing Rows : {test_len:,} ({test_len/total_len*100:.2f}%)")
    print(f"Sum Verification (Train + Test == Total): {train_len + test_len == total_len} ({train_len + test_len:,} rows)")
    
    # 4. Stratification Verification Table
    print("\n--- STEP 4: STRATIFICATION & CLASS DISTRIBUTION COMPARISON ---")
    orig_counts = y.value_counts()
    orig_pcts = y.value_counts(normalize=True) * 100
    
    train_counts = y_train.value_counts()
    train_pcts = y_train.value_counts(normalize=True) * 100
    
    test_counts = y_test.value_counts()
    test_pcts = y_test.value_counts(normalize=True) * 100
    
    strat_df = pd.DataFrame({
        'Full Dataset Count': orig_counts,
        'Full %': orig_pcts.round(2),
        'Train Set Count': train_counts,
        'Train %': train_pcts.round(2),
        'Test Set Count': test_counts,
        'Test %': test_pcts.round(2)
    })
    print(strat_df)
    
    # 7. Sample Check
    print("\n--- STEP 7: SAMPLE CHECK (Train vs Test Samples) ---")
    print("\nTraining Set Sample (X_train & y_train):")
    for text, label in zip(X_train.head(3), y_train.head(3)):
        print(f"  [{label:<8}]: \"{text[:90]}...\"")
        
    print("\nTesting Set Sample (X_test & y_test):")
    for text, label in zip(X_test.head(3), y_test.head(3)):
        print(f"  [{label:<8}]: \"{text[:90]}...\"")

    # 8. Create notebooks/03_train_test_split.ipynb
    print("\n--- STEP 8: CREATING JUPYTER NOTEBOOK notebooks/03_train_test_split.ipynb ---")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 5 — Train / Test Split Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Split the cleaned review corpus (393,579 rows) into 80% Training and 20% Testing sets using stratified sampling to prevent class skew and data leakage.\n",
                    "\n",
                    "### Objectives:\n",
                    "1. Load `data/processed/reviews_cleaned.csv`\n",
                    "2. Define Features ($X$) and Target ($y$)\n",
                    "3. Perform `train_test_split` with `test_size=0.20`, `random_state=42`, `stratify=y`\n",
                    "4. Verify Stratification and Class Distribution\n",
                    "5. Explain Data Leakage Prevention Principles"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from pathlib import Path\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "\n",
                    "# Load cleaned dataset\n",
                    "data_path = Path('../data/processed/reviews_cleaned.csv')\n",
                    "df = pd.read_csv(data_path)\n",
                    "df['clean_text'] = df['clean_text'].fillna('')\n",
                    "\n",
                    "print(f'Cleaned Dataset Loaded: {len(df):,} rows')\n",
                    "display(df[['Score', 'Sentiment', 'clean_text']].head(3))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Define Features (X) and Target (y)\n",
                    "We use `clean_text` as our input feature feature matrix $X$ and `Sentiment` as target vector $y$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "X = df['clean_text']\n",
                    "y = df['Sentiment']\n",
                    "print(f'X shape: {X.shape}, y shape: {y.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Perform Train / Test Split\n",
                    "We split into **80% Train** (314,863 samples) and **20% Test** (78,716 samples).\n",
                    "* `random_state=42`: Guarantees reproducible results across environments.\n",
                    "* `stratify=y`: Preserves exact 77.9% Positive / 14.5% Negative / 7.6% Neutral class ratios in both splits."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
                    ")\n",
                    "\n",
                    "print(f'Training set size : {len(X_train):,} ({len(X_train)/len(df)*100:.2f}%)')\n",
                    "print(f'Testing set size  : {len(X_test):,} ({len(X_test)/len(df)*100:.2f}%)')\n",
                    "print(f'Sum Verification  : {len(X_train) + len(X_test) == len(df)}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Stratification & Distribution Verification\n",
                    "Comparing sentiment class percentages across Full Dataset, Train Set, and Test Set."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "dist_df = pd.DataFrame({\n",
                    "    'Full Dataset %': (y.value_counts(normalize=True)*100).round(2),\n",
                    "    'Train Set %': (y_train.value_counts(normalize=True)*100).round(2),\n",
                    "    'Test Set %': (y_test.value_counts(normalize=True)*100).round(2)\n",
                    "})\n",
                    "display(dist_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Data Leakage Prevention Rule\n",
                    "> [!IMPORTANT]\n",
                    "> **CRITICAL NLP RULE**: TF-IDF vectorizers MUST NOT be fitted on the combined dataset or test dataset.\n",
                    "> * In Phase 6, we will run `fit_transform()` ONLY on `X_train` to learn vocabulary and Inverse Document Frequencies (IDF).\n",
                    "> * `X_test` will ONLY be transformed using `tfidf.transform(X_test)`."
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
    
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    print(f"Created notebook: {nb_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase5()
