# Customer Feedback Intelligence: Project Architecture

This document provides a comprehensive technical breakdown of the system architecture, machine learning prediction pipeline, aspect-based sentiment analysis (ABSA) engine, web backends, and data processing workflows.

---

## 1. System Architecture Overview

The system is designed as a modular, end-to-end NLP & Machine Learning platform for processing customer feedback reviews.

```
+-------------------------------------------------------------------------+
|                              DATA CORPUS                                |
|             Amazon Fine Food Reviews (393,579 Clean Reviews)            |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         PREPROCESSING PIPELINE                          |
|   HTML Removal -> Lowercasing -> Negation Preservation -> Tokenization  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                        FEATURE EXTRACTION ENGINE                        |
|              TF-IDF Vectorizer (ngram_range=(1,2), max_features=20000)  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      DEPLOYMENT MODEL INFERENCE                         |
|           Logistic Regression Classifier (Macro F1: 67.04%)             |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   ASPECT-BASED SENTIMENT ANALYSIS (ABSA)                |
|      Domain Topic Matching (Taste, Delivery, Price, Packaging, etc.)    |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                           DEPLOYMENT INTERFACES                         |
|  1. Streamlit Dashboard (app/app.py)                                    |
|  2. Flask REST API (backend/app.py) + Custom SPA (frontend/index.html)   |
+-------------------------------------------------------------------------+
```

---

## 2. Directory & Module Responsibilities

```
customer-feedback-intelligence/
├── app/
│   └── app.py                     # Streamlit frontend dashboard (Phase 12)
├── backend/
│   └── app.py                     # Flask REST API backend (Phase 12B)
├── frontend/
│   ├── index.html                 # Custom HTML5 SPA interface
│   ├── css/style.css              # Custom SaaS responsive design system
│   └── js/app.js                  # Vanilla JS SPA & fetch() API client
├── src/
│   ├── preprocessing.py           # Text cleaning & negation preservation logic
│   ├── aspect_analysis.py         # ABSA domain topic extraction rules
│   └── predict.py                 # Unified inference pipeline (predict_review)
├── models/
│   ├── tfidf_vectorizer.pkl       # Fitted TF-IDF Vectorizer (20,000 features)
│   ├── logistic_regression_model.pkl # Deployment Model (Logistic Regression)
│   ├── naive_bayes_model.pkl      # Baseline Model (MultinomialNB)
│   └── linear_svm_model.pkl       # Baseline Model (LinearSVC)
├── results/
│   ├── model_comparison.csv       # Multi-metric model evaluation statistics
│   ├── aspect_summary.csv         # Domain aspect frequency & sentiment summary
│   └── error_analysis_samples.csv # Misclassification diagnostic samples
├── docs/
│   ├── flask_basics.md            # Beginner guide to Flask APIs & JSON flow
│   └── project_architecture.md    # System architecture & pipeline guide
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Version control exclusion rules
└── README.md                      # Primary project portfolio README
```

---

## 3. End-to-End Prediction Pipeline (`src/predict.py`)

When a customer review string is submitted for inference:

1. **Validation**: Inspects text for non-empty string structure and length boundaries.
2. **Text Preprocessing (`clean_text`)**:
   - Strips HTML tags (`<br />`, `<div>`).
   - Converts URLs to standard tokens.
   - Expands English contractions (`don't` $\to$ `do not`, `can't` $\to$ `cannot`).
   - Retains negation words (`not`, `never`, `no`, `neither`, `nor`, `barely`, `hardly`).
   - Standardizes punctuation and extra whitespace.
3. **TF-IDF Transformation**: Vectorizes cleaned text using the pre-fitted `models/tfidf_vectorizer.pkl` (20,000 max features, 1–2 n-grams).
4. **Sentiment Classification**: Passes the sparse vector through `models/logistic_regression_model.pkl` to compute:
   - Predicted class: `Positive`, `Neutral`, or `Negative`.
   - Class probabilities via `predict_proba()`.
5. **Aspect Extraction (`analyze_aspect_sentiment`)**:
   - Splits review into clause boundaries.
   - Scans clauses for keywords belonging to 7 food product aspects (`Taste`, `Packaging`, `Price`, `Ingredients`, `Delivery`, `Quality`, `Customer Support`).
   - Assigns clause-level sentiment using keyword markers.
6. **Structured Response**: Returns a unified Python dictionary containing `original_text`, `clean_text`, `sentiment`, `confidence`, `probabilities`, and `aspects`.

---

## 4. REST API & Web Interfaces

### Flask Backend (`backend/app.py`)
- `GET /api/health`: Validates loading of TF-IDF vectorizer and deployment model.
- `POST /api/predict`: Receives `{ "review": "..." }`, invokes `predict_review()`, and returns structured JSON output.
- `GET /api/analytics`: Reads `results/` CSV files to serve aggregate KPIs and aspect summaries.

### Frontend Integration (`frontend/`)
- Single Page Application (SPA) built with pure HTML5, custom CSS3, and Vanilla JavaScript.
- Uses standard `fetch()` API calls without heavy framework overhead.
- Interactive Chart.js graphs for visual feedback analytics.

### Streamlit Backup Interface (`app/app.py`)
- Python-native dashboard supporting live review inference and interactive dataset visualizations.
