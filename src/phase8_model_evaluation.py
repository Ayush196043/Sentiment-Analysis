import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)

def run_phase8():
    data_path = Path("data/processed/reviews_cleaned.csv")
    model_dir = Path("models")
    results_dir = Path("results")
    plots_dir = Path("notebooks/evaluation_plots")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set plot style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})
    
    print("=" * 75)
    print("PHASE 8 — MODEL EVALUATION & METRICS")
    print("=" * 75)
    
    # ---------------------------------------------------------
    # STEP 1: LOAD ARTIFACTS & DATASET SPLIT
    # ---------------------------------------------------------
    print("\n--- STEP 1: LOADING DATASET & PRE-TRAINED MODEL ARTIFACTS ---")
    df = pd.read_csv(data_path)
    df['clean_text'] = df['clean_text'].fillna("")
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['clean_text'], df['Sentiment'], test_size=0.20, random_state=42, stratify=df['Sentiment']
    )
    
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    nb_path = model_dir / "naive_bayes_model.pkl"
    lr_path = model_dir / "logistic_regression_model.pkl"
    svm_path = model_dir / "linear_svm_model.pkl"
    
    tfidf = joblib.load(vectorizer_path)
    nb_model = joblib.load(nb_path)
    lr_model = joblib.load(lr_path)
    svm_model = joblib.load(svm_path)
    
    X_test_tfidf = tfidf.transform(X_test_raw)
    print(f"X_test_tfidf Shape: {X_test_tfidf.shape} (78,716 samples × 20,000 features)")
    
    labels_order = ['Negative', 'Neutral', 'Positive']
    
    # Models dictionary
    models = {
        'Multinomial Naive Bayes': nb_model,
        'Logistic Regression': lr_model,
        'Linear SVM (LinearSVC)': svm_model
    }
    
    metrics_summary = []
    reports_dict = {}
    cm_dict = {}
    
    # ---------------------------------------------------------
    # STEP 2 & 5: CALCULATE MULTI-METRICS & CLASSIFICATION REPORTS
    # ---------------------------------------------------------
    for name, model in models.items():
        print(f"\n" + "=" * 70)
        print(f"EVALUATING MODEL: {name}")
        print("=" * 70)
        
        y_pred = model.predict(X_test_tfidf)
        
        # Overall Accuracy
        acc = accuracy_score(y_test, y_pred) * 100
        
        # Macro Metrics
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Weighted Metrics
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Per-class Metrics
        p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_test, y_pred, labels=labels_order)
        
        # Print Classification Report
        report_str = classification_report(y_test, y_pred, labels=labels_order, digits=4)
        print(report_str)
        reports_dict[name] = report_str
        
        # Calculate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
        cm_dict[name] = cm
        
        # Plot & Save Confusion Matrix
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order, ax=ax)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Sentiment Label", fontsize=11, fontweight='bold')
        ax.set_ylabel("Actual Sentiment Label", fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        plot_filename = f"cm_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig.savefig(plots_dir / plot_filename, dpi=300)
        plt.close()
        
        metrics_summary.append({
            'Model': name,
            'Accuracy (%)': round(acc, 2),
            'Macro Precision (%)': round(p_macro * 100, 2),
            'Macro Recall (%)': round(r_macro * 100, 2),
            'Macro F1-Score (%)': round(f1_macro * 100, 2),
            'Weighted F1-Score (%)': round(f1_weighted * 100, 2),
            'Negative F1 (%)': round(f1_class[0] * 100, 2),
            'Neutral F1 (%)': round(f1_class[1] * 100, 2),
            'Positive F1 (%)': round(f1_class[2] * 100, 2)
        })

    # ---------------------------------------------------------
    # STEP 6: MODEL COMPARISON TABLE & EXPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 6: FINAL MODEL METRIC COMPARISON TABLE")
    print("=" * 75)
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_string(index=False))
    
    # Save comparison CSV
    summary_csv_path = results_dir / "model_comparison.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved structured metric comparison to: {summary_csv_path.resolve()}")
    
    # ---------------------------------------------------------
    # STEP 9: VISUAL MODEL COMPARISON CHART
    # ---------------------------------------------------------
    print("\n--- STEP 9: GENERATING VISUAL MODEL COMPARISON CHART ---")
    fig, ax = plt.subplots(figsize=(9, 5))
    x_indices = np.arange(len(summary_df))
    bar_width = 0.35
    
    rects1 = ax.bar(x_indices - bar_width/2, summary_df['Accuracy (%)'], bar_width, label='Accuracy (%)', color='#3498db')
    rects2 = ax.bar(x_indices + bar_width/2, summary_df['Macro F1-Score (%)'], bar_width, label='Macro F1-Score (%)', color='#e67e22')
    
    ax.set_title("Model Comparison: Overall Accuracy vs. Macro F1-Score", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Score (%)", fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(summary_df['Model'], fontsize=11, fontweight='bold')
    ax.set_ylim(50, 100)
    ax.legend(fontsize=11)
    
    for p in rects1:
        height = p.get_height()
        ax.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2., height + 0.8), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    for p in rects2:
        height = p.get_height()
        ax.annotate(f"{height:.2f}%", (p.get_x() + p.get_width() / 2., height + 0.8), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(plots_dir / "04_model_accuracy_f1_comparison.png", dpi=300)
    plt.close()
    print(f"Saved comparison chart to: {(plots_dir / '04_model_accuracy_f1_comparison.png').resolve()}")

    # ---------------------------------------------------------
    # STEP 11: CREATE JUPYTER NOTEBOOK notebooks/06_model_evaluation.ipynb
    # ---------------------------------------------------------
    print("\n--- STEP 11: CREATING NOTEBOOK notebooks/06_model_evaluation.ipynb ---")
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 8 — Model Evaluation & Metrics Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Perform rigorous multi-metric evaluation across all 3 trained models (Accuracy, Precision, Recall, F1, Macro F1, Confusion Matrices).\n",
                    "\n",
                    "### Key Metrics Covered:\n",
                    "* **Accuracy**: Overall fraction of correct predictions.\n",
                    "* **Precision**: Fraction of predicted positive instances that are actually positive.\n",
                    "* **Recall**: Fraction of actual positive instances correctly captured.\n",
                    "* **Macro F1-Score**: Equal-weighted arithmetic mean of F1 scores across Positive, Neutral, and Negative classes."
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
                    "from sklearn.metrics import classification_report, confusion_matrix\n",
                    "\n",
                    "# Load pre-trained models and test features\n",
                    "df = pd.read_csv('../data/processed/reviews_cleaned.csv')\n",
                    "df['clean_text'] = df['clean_text'].fillna('')\n",
                    "tfidf = joblib.load('../models/tfidf_vectorizer.pkl')\n",
                    "lr_model = joblib.load('../models/logistic_regression_model.pkl')\n",
                    "\n",
                    "_, X_test_raw, _, y_test = train_test_split(\n",
                    "    df['clean_text'], df['Sentiment'], test_size=0.20, random_state=42, stratify=df['Sentiment']\n",
                    ")\n",
                    "X_test_tfidf = tfidf.transform(X_test_raw)\n",
                    "\n",
                    "print(f'Test matrix shape: {X_test_tfidf.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Classification Report — Logistic Regression"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "y_pred = lr_model.predict(X_test_tfidf)\n",
                    "labels = ['Negative', 'Neutral', 'Positive']\n",
                    "print(classification_report(y_test, y_pred, labels=labels, digits=4))\n",
                    "print('Confusion Matrix:')\n",
                    "print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels))"
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
    
    nb_path = Path("notebooks/06_model_evaluation.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {nb_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase8()
