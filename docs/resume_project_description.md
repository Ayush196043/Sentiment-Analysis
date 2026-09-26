# Resume Project Description & Technical Stack

This document provides ready-to-use resume bullet points, project summaries, technical stack definitions, and safety guidelines for presenting the **Customer Feedback Intelligence** project on your resume and portfolio.

---

## 1. Executive Project Summary

- **Project Name**: Customer Feedback Intelligence
- **Problem**: E-commerce platforms receive high volumes of unstructured customer feedback. Manual analysis is slow, and overall star ratings mask underlying operational root causes (e.g., great product taste vs. damaged packaging).
- **Solution**: Built an end-to-end NLP and Machine Learning system that classifies 3-class overall sentiment (Positive, Neutral, Negative) and performs sentence-level Aspect-Based Sentiment Analysis (ABSA) across 7 key product domain topics. Served via a RESTful Flask API with a custom responsive web SPA and a Streamlit analytics dashboard.

---

## 2. Professional Resume Bullet Points

### Option A: Standard Full-Stack ML / NLP Bullet Points
* **Developed an end-to-end Customer Feedback Intelligence System** analyzing 393,579 Amazon reviews using NLP, Scikit-learn, and custom Aspect-Based Sentiment Analysis (ABSA) across 7 product domain topics.
* **Engineered a negation-preserving NLP preprocessing pipeline** and TF-IDF feature extractor ($1\text{--}2$ n-grams, 20,000 max features), handling 174,875 text duplicates to prevent data leakage between train and test splits.
* **Evaluated Multinomial Naive Bayes, Logistic Regression, and Linear SVM classifiers** using stratified cross-validation; selected Logistic Regression achieving **88.12% accuracy** and a **67.04% Macro F1-score** on 78,716 unseen test reviews.
* **Deployed a modular prediction microservice** via a Flask REST API and built two interactive user interfaces (a custom HTML5/CSS3/Vanilla JS single-page application and a Streamlit dashboard) for real-time review inference and visual feedback analytics.

### Option B: Concise 3-Bullet Resume Version
* **Built an AI Customer Feedback Intelligence platform** using Python, Scikit-learn, and TF-IDF to classify 3-class review sentiment and extract aspect-level feedback across 393k+ reviews.
* **Compared 3 baseline ML models** (Naive Bayes, Logistic Regression, Linear SVM) using Macro F1 to address class imbalance (77.9% positive), selecting Logistic Regression (**88.12% accuracy, 67.04% Macro F1**).
* **Integrated inference engine into a RESTful Flask API** supporting sentence-level ABSA clause matching, served to a responsive custom SPA and Streamlit dashboard.

---

## 3. Project Technical Stack

- **Core Language**: Python 3.11
- **Data Engineering & Manipulation**: Pandas, NumPy
- **Machine Learning & NLP**: Scikit-learn, NLTK, Joblib, Regex
- **Feature Extraction**: TF-IDF Vectorizer ($1\text{--}2$ n-grams, 20,000 vocabulary)
- **Algorithms Evaluated**: Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Classifier (LinearSVC)
- **Web Backend & APIs**: Flask, Flask-CORS, REST API (JSON)
- **Frontend & Visualizations**: HTML5, Custom CSS3, Vanilla JavaScript, Chart.js, Plotly, Streamlit
- **Development & Analysis**: Jupyter Notebooks, Git, Markdown

---

## 4. Resume Safety & Truthfulness Guidelines

To ensure professional credibility during background checks and technical interviews, follow these rules:

| DO Claim | DO NOT Claim |
|---|---|
| ✅ 393,579 clean, deduplicated dataset samples | ❌ Millions of live production users |
| ✅ Logistic Regression Macro F1 of **67.04%** and **88.12% Accuracy** | ❌ "100% accuracy" or "perfect sentiment detection" |
| ✅ Rule/keyword baseline ABSA engine | ❌ BERT / Transformer / LLM deep learning models |
| ✅ Flask REST API verified programmatically via test clients | ❌ Cloud production deployment on AWS/GCP (unless deployed) |
| ✅ Rating-derived proxy target labels (1–2 Neg, 3 Neu, 4–5 Pos) | ❌ Manually annotated gold-standard sentiment labels |
