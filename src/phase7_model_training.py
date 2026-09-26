import pandas as pd
import numpy as np
import time
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def run_phase7():
    input_path = Path("data/processed/reviews_cleaned.csv")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    nb_dir = Path("notebooks")
    
    print("=" * 75)
    print("PHASE 7 — MODEL TRAINING & PRELIMINARY COMPARISON")
    print("=" * 75)
    
    # 1. Load Data & TF-IDF Vectorizer
    print("\n--- STEP 1: LOADING DATASET & RE-CREATING EXACT PHASE 6 FEATURES ---")
    df = pd.read_csv(input_path)
    df['clean_text'] = df['clean_text'].fillna("")
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['clean_text'], df['Sentiment'], test_size=0.20, random_state=42, stratify=df['Sentiment']
    )
    
    # Load pre-fitted vectorizer from Phase 6
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    if vectorizer_path.exists():
        print(f"Loading pre-fitted vectorizer from: {vectorizer_path.resolve()}")
        tfidf = joblib.load(vectorizer_path)
    else:
        print("Fitting TF-IDF vectorizer (max_features=20000, ngram_range=(1,2))...")
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        tfidf.fit(X_train_raw)
        joblib.dump(tfidf, vectorizer_path)
        
    X_train_tfidf = tfidf.transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    print(f"X_train_tfidf Shape: {X_train_tfidf.shape}")
    print(f"X_test_tfidf Shape : {X_test_tfidf.shape}")
    
    results = []
    
    # ---------------------------------------------------------
    # STEP 2: MULTINOMIAL NAIVE BAYES
    # ---------------------------------------------------------
    print("\n--- STEP 2: TRAINING MULTINOMIAL NAIVE BAYES ---")
    nb_model = MultinomialNB()
    
    start_time = time.time()
    nb_model.fit(X_train_tfidf, y_train)
    nb_train_time = time.time() - start_time
    
    nb_preds = nb_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_preds) * 100
    
    joblib.dump(nb_model, model_dir / "naive_bayes_model.pkl")
    print(f"Multinomial Naive Bayes Trained in {nb_train_time:.2f}s | Accuracy: {nb_acc:.2f}%")
    results.append({
        "Model": "Multinomial Naive Bayes",
        "Training Time (s)": round(nb_train_time, 2),
        "Accuracy (%)": round(nb_acc, 2),
        "Saved Path": "models/naive_bayes_model.pkl"
    })
    
    # ---------------------------------------------------------
    # STEP 3: LOGISTIC REGRESSION
    # ---------------------------------------------------------
    print("\n--- STEP 3: TRAINING LOGISTIC REGRESSION ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    
    start_time = time.time()
    lr_model.fit(X_train_tfidf, y_train)
    lr_train_time = time.time() - start_time
    
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_preds) * 100
    
    joblib.dump(lr_model, model_dir / "logistic_regression_model.pkl")
    print(f"Logistic Regression Trained in {lr_train_time:.2f}s | Accuracy: {lr_acc:.2f}%")
    results.append({
        "Model": "Logistic Regression",
        "Training Time (s)": round(lr_train_time, 2),
        "Accuracy (%)": round(lr_acc, 2),
        "Saved Path": "models/logistic_regression_model.pkl"
    })
    
    # ---------------------------------------------------------
    # STEP 4: LINEAR SVM
    # ---------------------------------------------------------
    print("\n--- STEP 4: TRAINING LINEAR SVM (LinearSVC) ---")
    svm_model = LinearSVC(max_iter=2000, random_state=42)
    
    start_time = time.time()
    svm_model.fit(X_train_tfidf, y_train)
    svm_train_time = time.time() - start_time
    
    svm_preds = svm_model.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_preds) * 100
    
    joblib.dump(svm_model, model_dir / "linear_svm_model.pkl")
    print(f"Linear SVM Trained in {svm_train_time:.2f}s | Accuracy: {svm_acc:.2f}%")
    results.append({
        "Model": "Linear SVM (LinearSVC)",
        "Training Time (s)": round(svm_train_time, 2),
        "Accuracy (%)": round(svm_acc, 2),
        "Saved Path": "models/linear_svm_model.pkl"
    })
    
    # Save predictions array dictionary for Phase 8 evaluation script
    preds_dict = {
        'y_test': list(y_test),
        'y_pred_nb': list(nb_preds),
        'y_pred_lr': list(lr_preds),
        'y_pred_svm': list(svm_preds)
    }
    with open("data/processed/model_predictions.json", "w") as f:
        json.dump(preds_dict, f)
        
    # ---------------------------------------------------------
    # STEP 7: PRELIMINARY COMPARISON TABLE
    # ---------------------------------------------------------
    print("\n--- STEP 7: PRELIMINARY COMPARISON TABLE ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # ---------------------------------------------------------
    # STEP 8: BUILD NOTEBOOK notebooks/05_model_training.ipynb
    # ---------------------------------------------------------
    print("\n--- STEP 8: CREATING NOTEBOOK notebooks/05_model_training.ipynb ---")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 7 — Model Training & Comparison Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Train and compare three baseline machine learning algorithms on 20,000 TF-IDF features.\n",
                    "\n",
                    "### Classifiers Benchmark:\n",
                    "1. **Multinomial Naive Bayes** (Fast probabilistic baseline)\n",
                    "2. **Logistic Regression** (Linear probabilistic model with L2 regularization)\n",
                    "3. **Linear SVM (`LinearSVC`)** (Maximum margin hyperplane separator)"
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
                    "import time\n",
                    "from pathlib import Path\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.naive_bayes import MultinomialNB\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.svm import LinearSVC\n",
                    "from sklearn.metrics import accuracy_score\n",
                    "\n",
                    "# Load pre-fitted TF-IDF vectorizer and datasets\n",
                    "df = pd.read_csv('../data/processed/reviews_cleaned.csv')\n",
                    "df['clean_text'] = df['clean_text'].fillna('')\n",
                    "tfidf = joblib.load('../models/tfidf_vectorizer.pkl')\n",
                    "\n",
                    "X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n",
                    "    df['clean_text'], df['Sentiment'], test_size=0.20, random_state=42, stratify=df['Sentiment']\n",
                    ")\n",
                    "\n",
                    "X_train_tfidf = tfidf.transform(X_train_raw)\n",
                    "X_test_tfidf = tfidf.transform(X_test_raw)\n",
                    "\n",
                    "print(f'Train matrix: {X_train_tfidf.shape}')\n",
                    "print(f'Test matrix : {X_test_tfidf.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Multinomial Naive Bayes\n",
                    "Applies Bayes' Theorem assuming independence among TF-IDF feature frequencies."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "nb = MultinomialNB()\n",
                    "start = time.time()\n",
                    "nb.fit(X_train_tfidf, y_train)\n",
                    "nb_time = time.time() - start\n",
                    "nb_acc = accuracy_score(y_test, nb.predict(X_test_tfidf)) * 100\n",
                    "print(f'Naive Bayes Accuracy: {nb_acc:.2f}% ({nb_time:.2f}s)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Logistic Regression\n",
                    "Optimizes log-loss over high-dimensional sparse TF-IDF features."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "lr = LogisticRegression(max_iter=1000, random_state=42)\n",
                    "start = time.time()\n",
                    "lr.fit(X_train_tfidf, y_train)\n",
                    "lr_time = time.time() - start\n",
                    "lr_acc = accuracy_score(y_test, lr.predict(X_test_tfidf)) * 100\n",
                    "print(f'Logistic Regression Accuracy: {lr_acc:.2f}% ({lr_time:.2f}s)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Linear SVM (LinearSVC)\n",
                    "Finds the maximum margin hyper-plane separating Positive, Neutral, and Negative clusters."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "svm = LinearSVC(max_iter=2000, random_state=42)\n",
                    "start = time.time()\n",
                    "svm.fit(X_train_tfidf, y_train)\n",
                    "svm_time = time.time() - start\n",
                    "svm_acc = accuracy_score(y_test, svm.predict(X_test_tfidf)) * 100\n",
                    "print(f'Linear SVM Accuracy: {svm_acc:.2f}% ({svm_time:.2f}s)')"
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
    
    nb_path = nb_dir / "05_model_training.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {nb_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase7()
