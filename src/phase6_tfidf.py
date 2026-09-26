import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def run_phase6():
    input_path = Path("data/processed/reviews_cleaned.csv")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    nb_path = Path("notebooks/04_tfidf.ipynb")
    
    print("=" * 75)
    print("PHASE 6 — TF-IDF FEATURE EXTRACTION")
    print("=" * 75)
    
    # 1. Load Cleaned Dataset & Split
    print("\n--- STEP 1: LOADING & SPLITTING CLEANED DATASET ---")
    df = pd.read_csv(input_path)
    df['clean_text'] = df['clean_text'].fillna("")
    
    X = df['clean_text']
    y = df['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"X_train Size: {len(X_train):,} samples")
    print(f"X_test Size : {len(X_test):,} samples")
    
    # 4 & 5. Create & Fit TF-IDF Vectorizer (Train Only!)
    print("\n--- STEP 4 & 5: CREATING & FITTING TF-IDF VECTORIZER (X_train ONLY) ---")
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )
    
    print("Fitting TfidfVectorizer on X_train and transforming X_train...")
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    print("Transforming X_test (Strictly using pre-fitted vectorizer)...")
    X_test_tfidf = tfidf.transform(X_test)
    
    # 6. Inspect Matrix Shapes
    print("\n--- STEP 6: TF-IDF MATRIX DIMENSIONS & SPARSITY ---")
    print(f"X_train_tfidf Shape: {X_train_tfidf.shape} (314,863 documents × 20,000 features)")
    print(f"X_test_tfidf Shape : {X_test_tfidf.shape}  (78,716 documents × 20,000 features)")
    
    sparsity_train = 100.0 * (1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))
    print(f"Non-zero elements in X_train_tfidf: {X_train_tfidf.nnz:,}")
    print(f"Matrix Sparsity (% of zeroes)     : {sparsity_train:.2f}%")
    print(f"Sparse Memory Usage              : {X_train_tfidf.data.nbytes / (1024**2):.2f} MB (Sparse CSR Format)")
    
    # 7. Feature Inspection
    print("\n--- STEP 7: LEARNED FEATURE INSPECTION ---")
    feature_names = tfidf.get_feature_names_out()
    print(f"Total Features Learned: {len(feature_names):,}")
    print(f"First 15 Unigrams & Bigrams  : {list(feature_names[:15])}")
    
    # Search for key sentiment & negation features
    sample_key_features = ['good', 'bad', 'not good', 'not worth', 'great', 'terrible', 'never buy', 'highly recommend']
    found_features = [f for f in sample_key_features if f in feature_names]
    print(f"Sample Key Features Found    : {found_features}")
    
    # 10. Save Fitted Vectorizer
    print("\n--- STEP 10: SAVING SAVED VECTORIZER ---")
    joblib.dump(tfidf, vectorizer_path)
    print(f"Saved fitted TF-IDF vectorizer to: {vectorizer_path.resolve()}")
    
    # 11. Create Notebook notebooks/04_tfidf.ipynb
    print("\n--- STEP 11: CREATING NOTEBOOK notebooks/04_tfidf.ipynb ---")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 6 — TF-IDF Feature Extraction Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Convert clean review text into numerical TF-IDF feature matrices for machine learning algorithms.\n",
                    "\n",
                    "### Objectives:\n",
                    "1. Load `X_train` (314,863 samples) and `X_test` (78,716 samples)\n",
                    "2. Explain TF-IDF Theory (Term Frequency & Inverse Document Frequency)\n",
                    "3. Configure `TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)`\n",
                    "4. Fit vectorizer strictly on `X_train` (`fit_transform`)\n",
                    "5. Transform `X_test` (`transform`) to prevent Data Leakage\n",
                    "6. Inspect Matrix Shapes, Sparsity, and Learned Features\n",
                    "7. Save fitted vectorizer to `models/tfidf_vectorizer.pkl`"
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
                    "import joblib\n",
                    "from pathlib import Path\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                    "\n",
                    "# Load cleaned dataset\n",
                    "df = pd.read_csv('../data/processed/reviews_cleaned.csv')\n",
                    "df['clean_text'] = df['clean_text'].fillna('')\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    df['clean_text'], df['Sentiment'], test_size=0.20, random_state=42, stratify=df['Sentiment']\n",
                    ")\n",
                    "\n",
                    "print(f'X_train samples: {len(X_train):,}')\n",
                    "print(f'X_test samples : {len(X_test):,}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. TF-IDF Concept & Formula\n",
                    "$$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\text{IDF}(t)$$\n",
                    "* **Term Frequency (TF)**: Relative frequency of word $t$ in document $d$.\n",
                    "* **Inverse Document Frequency (IDF)**: Penalizes words that appear across almost all documents (e.g. `the`, `is`, `product`) and boosts domain-specific sentiment terms (`excellent`, `terrible`, `not good`)."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Fit Vectorizer (Train Only) & Transform Test\n",
                    "> [!IMPORTANT]\n",
                    "> **DATA LEAKAGE SAFETY**: Vectorizer is fitted using `fit_transform()` ONLY on `X_train`. `X_test` is transformed using `transform()`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "tfidf = TfidfVectorizer(\n",
                    "    max_features=20000,\n",
                    "    ngram_range=(1, 2),\n",
                    "    min_df=2\n",
                    ")\n",
                    "\n",
                    "X_train_tfidf = tfidf.fit_transform(X_train)\n",
                    "X_test_tfidf = tfidf.transform(X_test)\n",
                    "\n",
                    "print(f'X_train_tfidf shape: {X_train_tfidf.shape}')\n",
                    "print(f'X_test_tfidf shape : {X_test_tfidf.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Feature Inspection & Sparse Matrix Check"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "features = tfidf.get_feature_names_out()\n",
                    "print(f'Total Features Learned: {len(features):,}')\n",
                    "print('Sample Unigrams & Bigrams:', list(features[5000:5015]))\n",
                    "\n",
                    "# Save vectorizer\n",
                    "joblib.dump(tfidf, '../models/tfidf_vectorizer.pkl')\n",
                    "print('Vectorizer saved to models/tfidf_vectorizer.pkl')"
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
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    print(f"Created notebook: {nb_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase6()
