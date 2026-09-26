# Complete Project Explanation & Interview Walkthrough

This document provides step-by-step explanations, elevator pitches, and deep-dive technical walkthroughs for presenting the **Customer Feedback Intelligence** project during job interviews and technical presentations.

---

## 1. System Architecture Step-by-Step

```
[1. Raw Dataset]
       ↓ (568,454 original Amazon reviews)
[2. Data Quality / EDA]
       ↓ (Removed 174,875 text duplicates -> 393,579 clean samples)
[3. NLP Preprocessing]
       ↓ (HTML removal, contraction expansion, negation preservation)
[4. Stratified Train/Test Split]
       ↓ (80% Train: 314,863 | 20% Test: 78,716)
[5. TF-IDF Vectorization]
       ↓ (Fitted ONLY on X_train: 20,000 max features, 1-2 n-grams)
[6. ML Model Training]
       ↓ (MultinomialNB, Logistic Regression, LinearSVC)
[7. Multi-Metric Evaluation]
       ↓ (Macro F1 selection: Logistic Regression 67.04% Macro F1)
[8. Error Analysis]
       ↓ (Inspected 9,348 test errors: Neutral ambiguity & length bias)
[9. Aspect-Based Sentiment Analysis]
       ↓ (Rule-based clause extraction for 7 product topics)
[10. Saved Deployment Artifacts]
       ↓ (models/tfidf_vectorizer.pkl & models/logistic_regression_model.pkl)
[11. Unified Prediction Pipeline]
       ↓ (src/predict.py -> predict_review())
[12. Flask REST API & Web Interfaces]
       ↓ (backend/app.py -> GET /api/health, POST /api/predict, GET /api/analytics)
[13. Frontends]
       ↓ (Custom HTML/CSS/JS SPA & Streamlit Dashboard app/app.py)
```

---

## 2. 30-Second Project Introduction (Elevator Pitch)

> *"For my project, I built an **AI Customer Feedback Intelligence System** that analyzes unstructured customer reviews to extract overall sentiment as well as aspect-specific feedback. Using the Amazon Fine Food Reviews dataset of over 393,000 clean reviews, I engineered a negation-preserving NLP preprocessing pipeline and compared Naive Bayes, Logistic Regression, and Linear SVM models. 
> 
> Because the dataset is heavily imbalanced toward positive reviews, I prioritized **Macro F1-Score** over plain accuracy and selected Logistic Regression, achieving an **88.12% accuracy and 67.04% Macro F1**. I also built a rule-based Aspect-Based Sentiment Analysis (ABSA) module to isolate complaints like bad delivery or damaged packaging from product quality, and deployed the entire pipeline via a RESTful Flask API serving a custom responsive web interface."*

---

## 3. 2-Minute Comprehensive Interview Deep-Dive

> *"Let me give you a detailed walkthrough of how I built the Customer Feedback Intelligence platform.
> 
> **The Problem & Dataset**: E-commerce companies receive massive amounts of customer feedback, but reading reviews manually is unscalable, and overall star ratings mask operational issues—for instance, a customer might love a product's taste but hate its late delivery. I used the Amazon Fine Food Reviews dataset containing 568,454 reviews. During data quality checks, I discovered 174,875 exact duplicate texts created by cross-posted products. I deduplicated these to prevent data leakage between training and testing sets, resulting in 393,579 unique reviews.
> 
> **Label Creation & Preprocessing**: I constructed 3 target sentiment classes from star ratings: 1–2 stars as Negative, 3 stars as Neutral, and 4–5 stars as Positive. In preprocessing, standard stopword removal destroys crucial sentiment negation—words like 'not' or 'never'. I implemented custom regular expressions to expand contractions like 'don't' to 'do not' and preserved all negation tokens.
> 
> **Feature Extraction & Modeling**: I performed an 80/20 stratified train-test split and fitted a TF-IDF vectorizer strictly on the training set using 1–2 n-grams and 20,000 max features. I evaluated three classifiers: Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine. Because 77.9% of reviews were Positive, accuracy was misleading. Using **Macro F1-Score**, Logistic Regression outperformed the rest with **67.04% Macro F1 and 88.12% overall accuracy** on 78,716 unseen test reviews.
> 
> **Error Analysis & ABSA**: Error analysis on 9,348 misclassifications revealed that 3-star Neutral reviews accounted for 76.7% of errors due to mixed sentiment text. To solve the mixed sentiment challenge, I developed a sentence-level ABSA module targeting 7 food domain topics like Taste, Packaging, Price, and Delivery. This allowed the system to output 'Taste: Positive' and 'Delivery: Negative' for a single review.
> 
> **Deployment Architecture**: Finally, I packaged the vectorizer and model into a unified Python inference engine (`src/predict.py`) and built a RESTful Flask API (`backend/app.py`). I developed two frontends: a custom HTML/CSS/JS Single Page Application and a Streamlit dashboard."*

---

## 4. Honest Limitations (Interview-Ready)

1. **Proxy Label Noise**: Target labels are derived from 1–5 star ratings rather than human-annotated sentiment text. Text/rating mismatches (e.g., polite phrasing with a 1-star rating) exist in the data.
2. **Historical Corpus**: The dataset covers reviews up to 2012; modern slang, e-commerce vocabulary, and consumer trends differ today.
3. **Class Imbalance & Neutral Difficulty**: Positive reviews dominate (77.94%), while Neutral 3-star reviews suffer from low precision/recall (Neutral F1 is 32.06%).
4. **Keyword Baseline ABSA**: The current ABSA system uses rule-based keyword and clause matching, which struggles with complex syntactic dependencies compared to Transformer architectures.
5. **Local Browser Verification Status**: Flask REST API endpoints were programmatically verified via test clients (`test_client()`), but interactive local browser testing is currently pending manual check.

---

## 5. Realistic Future Improvements

- **Transformer Finetuning**: Implement BERT or RoBERTa for context-aware sentiment classification.
- **Deep Learning ABSA**: Migrate to PyABSA or DeBERTa aspect-level extraction.
- **Human-Annotated Test Set**: Benchmark the system against a gold-standard human-labeled dataset.
- **Cloud Microservice Deployment**: Containerize with Docker and deploy to AWS / GCP with CI/CD monitoring.
