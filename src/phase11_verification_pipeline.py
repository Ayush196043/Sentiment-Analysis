import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append("src")
from predict import predict_review, tfidf_vectorizer, sentiment_model

def run_phase11():
    model_dir = Path("models")
    nb_dir = Path("notebooks")
    
    print("=" * 75)
    print("PHASE 11 — PREDICTION PIPELINE VERIFICATION")
    print("=" * 75)
    
    # ---------------------------------------------------------
    # STEP 1: VERIFY SAVED ARTIFACTS
    # ---------------------------------------------------------
    print("\n--- STEP 1: VERIFYING SAVED ARTIFACTS ---")
    artifacts = [
        "models/tfidf_vectorizer.pkl",
        "models/logistic_regression_model.pkl",
        "models/naive_bayes_model.pkl",
        "models/linear_svm_model.pkl",
        "src/preprocessing.py",
        "src/aspect_analysis.py",
        "src/predict.py"
    ]
    for art in artifacts:
        p = Path(art)
        exists = p.exists()
        size_kb = (p.stat().st_size / 1024) if exists else 0
        print(f"  • {art:<40}: {'EXISTS' if exists else 'MISSING'} ({size_kb:.2f} KB)")
        
    # ---------------------------------------------------------
    # STEP 5 & 6: TEST WITH 5 REAL EXAMPLES
    # ---------------------------------------------------------
    print("\n--- STEP 5 & 6: TESTING PREDICTION PIPELINE WITH 5 REAL EXAMPLES ---")
    test_cases = [
        ("Clearly Positive", "The dog food quality is amazing and my Labrador loves it! Great price and fast delivery."),
        ("Clearly Negative", "Product arrived damaged, box was unsealed, and the peanuts were stale and rancid. Terrible experience."),
        ("Neutral / Ambiguous", "The product is okay, nothing special but not bad either."),
        ("Mixed Sentiment", "The taste is excellent and delicious, but the delivery was late and packaging was crushed."),
        ("Negation Handling", "This coffee is not worth the price and definitely not fresh.")
    ]
    
    for category, review in test_cases:
        res = predict_review(review)
        print(f"\n[{category.upper()}]")
        print(f"  Review Text: \"{res['original_text']}\"")
        print(f"  Clean Text : \"{res['clean_text']}\"")
        print(f"  Prediction : {res['sentiment']} (Confidence: {res['confidence']*100:.2f}%)")
        print(f"  Probs Dict : {res['probabilities']}")
        print(f"  Aspects ({len(res['aspects'])} detected):")
        for asp in res['aspects']:
            print(f"     -> Aspect: {asp['aspect']:<15} | Sentiment: {asp['sentiment']:<8} | Clause: \"{asp['matching_clause']}\"")

    # ---------------------------------------------------------
    # STEP 8: CREATE NOTEBOOK notebooks/09_prediction_pipeline.ipynb
    # ---------------------------------------------------------
    print("\n--- STEP 8: CREATING NOTEBOOK notebooks/09_prediction_pipeline.ipynb ---")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 11 — Inference & Prediction Pipeline Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Load pre-trained TF-IDF vectorizer and Logistic Regression deployment model to create a unified inference pipeline (`predict_review()`) with aspect extraction.\n",
                    "\n",
                    "### Inference Architecture:\n",
                    "```\n",
                    "Customer Review Text\n",
                    "        │\n",
                    "        ▼\n",
                    "Pre-training Preprocessing (clean_text)\n",
                    "        │\n",
                    "        ▼\n",
                    "TF-IDF Transform (tfidf_vectorizer.pkl)\n",
                    "        │\n",
                    "        ▼\n",
                    "Logistic Regression Inference (logistic_regression_model.pkl)\n",
                    "        │\n",
                    "        ├─────────────────────────────┐\n",
                    "        ▼                             ▼\n",
                    "Overall Sentiment & Confidence    Aspect-Based Sentiment Extraction\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "sys.path.append('../src')\n",
                    "from predict import predict_review\n",
                    "\n",
                    "sample = 'The taste is excellent but delivery was late and expensive.'\n",
                    "result = predict_review(sample)\n",
                    "print(f'Sentiment: {result[\"sentiment\"]} (Confidence: {result[\"confidence\"]*100:.2f}%)')\n",
                    "print('Probabilities:', result['probabilities'])\n",
                    "print('Aspect Insights:')\n",
                    "for asp in result['aspects']:\n",
                    "    print(f\"  • {asp['aspect']}: {asp['sentiment']} (Clause: '{asp['matching_clause']}')\")"
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
    
    nb_path = nb_dir / "09_prediction_pipeline.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {nb_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase11()
