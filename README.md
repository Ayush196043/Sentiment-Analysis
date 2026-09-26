# Customer Feedback Intelligence: AI-Powered Review Analytics & ABSA System

An end-to-end Natural Language Processing (NLP) and Machine Learning system designed to analyze customer reviews, classify sentiment, extract aspect-level feedback, and deliver business intelligence through interactive web applications.

---

## 1. Project Overview

**Customer Feedback Intelligence** processes large volumes of customer reviews to extract actionable insights. Beyond classifying overall sentiment as **Positive**, **Neutral**, or **Negative**, the system performs **Aspect-Based Sentiment Analysis (ABSA)** to pinpoint specific topic feedback (e.g., product taste, packaging quality, delivery speed, or pricing).

Key capabilities:
- **3-Class Sentiment Classification**: Classifies customer review text into Positive, Neutral, or Negative sentiment.
- **Aspect-Based Sentiment Analysis (ABSA)**: Identifies topic-specific sentiments within single or multi-sentence reviews.
- **Mixed Sentiment Handling**: Resolves complex reviews containing contrasting opinions across different aspects.
- **Dual Web Interfaces**: Features both a **Streamlit Dashboard** and a **Custom HTML/CSS/JS + Flask REST API** web application.

---

## 2. Problem Statement

E-commerce businesses receive thousands of customer reviews daily. Manually reviewing every submission is time-consuming and unscalable. Furthermore, overall star ratings often obscure critical operational issues. 

For example, a customer might leave a 2-star review stating:  
> *"The coffee tastes amazing, but the bag arrived damaged and delivery was two weeks late."*

A standard classifier or star rating labels the entire interaction as negative, hiding the fact that product quality is high while logistics and packaging need urgent operational attention. This project addresses that challenge by combining overall sentiment prediction with granular aspect extraction.

---

## 3. Key Features

- **Empirical Baseline Model Comparison**: Systematic evaluation of Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Classifier (LinearSVC).
- **Negation-Preserving NLP Preprocessing**: Custom cleaning pipeline that preserves critical negation context (`not good`, `never buy`).
- **Aspect Extraction Engine**: Baseline rule/keyword matching for 7 food product categories (*Taste*, *Packaging*, *Price*, *Ingredients*, *Delivery*, *Quality*, *Customer Support*).
- **Error Analysis**: Diagnostic investigation into misclassified reviews, length bias, and neutral class ambiguity.
- **Modular Production Pipeline**: Decoupled inference engine (`src/predict.py`) powering both web frontends.
- **RESTful Flask Backend**: Microservice API returning structured JSON predictions with model confidence probabilities.

---

## 4. System Architecture

### Machine Learning Pipeline Flow

```
Customer Review String
          │
          ▼
   Text Preprocessing
(HTML removal, lowercasing, contraction expansion, negation preservation)
          │
          ▼
   TF-IDF Feature Extraction
(1–2 n-grams, min_df=2, 20,000 max features)
          │
          ▼
   Logistic Regression Model
(Supervised 3-class prediction + confidence probabilities)
          │
          ▼
   Aspect-Based Sentiment Analysis (ABSA)
(Clause boundary splitting & domain keyword matching)
          │
          ▼
   Structured Prediction Output (JSON / UI Display)
```

### Custom Web Application Architecture

```
User Browser (HTML5 / CSS3 / Vanilla JS SPA)
          │
          ▼  HTTP REST Requests (fetch)
Flask API Server (backend/app.py)
          │
          ▼  Inference Function Call
Prediction Pipeline (src/predict.py)
          │
          ▼  Structured Dictionary
JSON API Response (HTTP 200)
          │
          ▼  Dynamic DOM Update
User Interface Display (Sentiment Badge + ABSA Cards + Chart.js)
```

---

## 5. Dataset

- **Corpus**: [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (SNAP / Kaggle).
- **Dataset Size**: 568,454 original raw reviews $\to$ **393,579 clean deduplicated reviews**.
- **Historical Snapshot**: Historical review snapshot covering October 1999 to October 2012.
- **Primary Fields Utilized**:
  - `Text`: Customer review body text (Primary feature input).
  - `Score`: Star rating 1 to 5 (Used for proxy label construction).

*Note: This dataset represents historical customer review data and should not be interpreted as a reflection of current Amazon customer sentiment or product offerings.*

---

## 6. Sentiment Label Creation

Sentiment labels were derived from customer star ratings (`Score`) using the following project mapping:

| Star Rating | Proxy Sentiment Label | Class Distribution (Dataset Count) | Class Percentage |
|---|---|---|---|
| **4 – 5 Stars** | **Positive** | 306,779 reviews | 77.94% |
| **1 – 2 Stars** | **Negative** | 57,031 reviews | 14.50% |
| **3 Stars** | **Neutral** | 29,769 reviews | 7.56% |

> [!IMPORTANT]
> **Proxy Label Disclaimer**: These target labels are project-defined proxy labels derived directly from numerical star ratings. They are **not** equivalent to manually annotated sentiment ground truth. Disagreements between written text and star ratings (e.g., sarcasm or polite text with low ratings) represent inherent dataset noise.

---

## 7. Data Processing Pipeline

1. **Deduplication**: Identified and removed 174,875 duplicate review texts created by repeat cross-postings.
2. **Missing Value Handling**: Verified zero null values in critical text and rating columns.
3. **Text Normalization (`src/preprocessing.py`)**:
   - Stripped raw HTML tags (`<br />`, `<div>`, etc.).
   - Standardized URLs to generic placeholders.
   - Expanded contractions (`don't` $\to$ `do not`, `won't` $\to$ `will not`).
   - Retained crucial negation words (`not`, `never`, `no`, `neither`, `nor`, `barely`, `hardly`).
   - Cleaned special symbols while keeping alphanumeric tokens and sentence boundaries.
4. **Stratified Train/Test Split**: 80% Training (314,863 samples) / 20% Testing (78,716 samples), preserving class proportions.
5. **TF-IDF Feature Extraction**: Fitted strictly on `X_train` to prevent data leakage (`ngram_range=(1,2)`, `max_features=20000`, `min_df=2`).

---

## 8. Model Training

Three machine learning models were trained on 314,863 vectorized training reviews:

1. **Multinomial Naive Bayes**: Fast probabilistic baseline for text classification.
2. **Logistic Regression (`L-BFGS`, `C=1.0`)**: Linear linear decision boundary optimizer with calibrated `predict_proba()` confidence distributions.
3. **Linear Support Vector Classifier (`LinearSVC`)**: Maximum-margin linear classifier optimized for high-dimensional sparse text vectors.

---

## 9. Model Evaluation & Results

Evaluated on **78,716 unseen test reviews**. Macro F1-Score was chosen as the primary metric due to significant dataset class imbalance (77.9% Positive).

### Empirical Performance Summary

| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1-Score (%) |
|---|---|---|---|---|---|
| **Multinomial Naive Bayes** | 84.81% | 72.60% | 52.74% | 56.05% | 81.60% |
| **Logistic Regression (Selected)** | **88.12%** | **73.48%** | **64.29%** | **67.04%** | **86.74%** |
| **Linear SVM (LinearSVC)** | 88.08% | 72.71% | 64.39% | 66.60% | 86.68% |

### Selected Model (Logistic Regression) Class-Wise F1 Breakdown

- **Positive Class F1**: **94.26%**
- **Negative Class F1**: **74.79%**
- **Neutral Class F1**: **32.06%** *(Reflects significant ambiguity in 3-star reviews)*

> [!NOTE]
> **Why Accuracy Alone Is Misleading**: A naive baseline predicting "Positive" for every review achieves ~77.9% accuracy. Macro F1 (67.04%) provides an honest measure of performance across minority Negative and Neutral classes.

---

## 10. Error Analysis

An inspection of **9,348 misclassified test reviews** revealed three primary error patterns:

1. **Neutral Class Ambiguity**: Neutral reviews accounted for 76.7% of all model errors. 3-star reviews frequently mix mild praise with mild criticism, confusing linear decision boundaries.
2. **Review Length Bias**: Misclassified reviews were **21.8% longer** on average than correctly classified reviews, introducing conflicting sentiment signals.
3. **Rating / Text Mismatches**: Samples where customers wrote highly positive text but selected 1-star ratings (or vice versa), representing noise inherent in star-rating proxy labels.

---

## 11. Aspect-Based Sentiment Analysis (ABSA)

The system includes a baseline rule/keyword ABSA engine (`src/aspect_analysis.py`) targeting 7 key food domain aspects:

- **Taste** (`flavor`, `delicious`, `bitter`, `sweet`, `yum`, etc.)
- **Packaging** (`box`, `can`, `bag`, `damaged`, `leaked`, `sealed`, etc.)
- **Price** (`expensive`, `cheap`, `cost`, `value`, `pricey`, etc.)
- **Ingredients** (`sugar`, `organic`, `natural`, `gluten`, `preservatives`, etc.)
- **Delivery** (`shipping`, `arrived`, `late`, `fast`, `delivered`, etc.)
- **Quality** (`fresh`, `stale`, `rotten`, `quality`, `premium`, etc.)
- **Customer Support** (`refund`, `seller`, `service`, `replacement`, etc.)

### Mixed Sentiment Example

> **Input Review**: *"The taste is excellent but delivery was terrible."*

- **Overall Sentiment**: `Negative` (or `Positive` depending on clause weights)
- **Aspect Extraction**:
  - `Taste` $\to$ **Positive** (*"The taste is excellent"*)
  - `Delivery` $\to$ **Negative** (*"delivery was terrible"*)

---

## 12. Prediction Pipeline (`src/predict.py`)

Inference is encapsulated in `src/predict.py`:

```python
from predict import predict_review

result = predict_review("The taste is amazing but packaging was broken.")
print(result["sentiment"])   # Overall sentiment
print(result["confidence"])  # Model probability score
print(result["aspects"])     # List of extracted aspect dictionaries
```

---

## 13. Web Applications

The project provides two complete frontend deployments:

### 1. Streamlit Dashboard (`app/app.py`)
Interactive multi-page dashboard featuring Overview KPI cards, Review Analyzer hero view, Analytics tab, Aspect Insights, and Project Info.

### 2. Custom HTML/CSS/JS + Flask API (`frontend/` + `backend/app.py`)
Custom Single Page Application (SPA) built with responsive CSS grid styling, Vanilla JavaScript, Chart.js graphs, and a Flask REST microservice API.

---

## 14. Visualizations & Evaluation Plots

Visualization artifacts generated during EDA and model evaluation:
- `notebooks/eda_plots/01_sentiment_distribution.png` — Dataset class imbalance donut plot.
- `notebooks/eda_plots/02_rating_distribution.png` — 1–5 Star rating counts.
- `notebooks/evaluation_plots/confusion_matrix_logistic_regression.png` — Normalized confusion matrix heatmap.

---

## 15. Installation & Setup

### Environment Setup (Windows)

```bash
# Clone or navigate to the project directory
cd "d:/Project For Placement/Sentiment analysis project"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 16. Running the Streamlit Application

To launch the Streamlit frontend:

```bash
streamlit run app/app.py
```

Access the Streamlit app at: `http://localhost:8501`

---

## 17. Running the Flask Application

To launch the Flask REST API & Custom Web Frontend:

```bash
python backend/app.py
```

Access the custom web dashboard at: `http://127.0.0.1:5000`

> [!NOTE]
> **Verification Status**: Flask REST API endpoints (`/api/health`, `/api/predict`, `/api/analytics`) have been verified programmatically via unit tests. Local interactive browser verification is pending manual user browser testing.

---

## 18. Project Limitations

- **Proxy Label Noise**: Target labels are derived from star ratings rather than manual sentiment annotations.
- **Historical Dataset**: Dataset snapshot covers reviews up to 2012; vocabulary and product trends may differ today.
- **Rule-Based ABSA**: Baseline keyword aspect extraction lacks deep semantic context compared to modern Transformer architectures.
- **Neutral Class Performance**: Low Neutral F1 (32.06%) due to subjective boundaries of 3-star ratings.
- **Sarcasm & Complex Context**: Bag-of-words TF-IDF representation struggles with subtle sarcasm or indirect language.

---

## 19. Future Improvements

- **Transformer Models**: Upgrade to fine-tuned BERT / RoBERTa / DistilBERT for context-aware sentiment classification.
- **Deep Learning ABSA**: Implement PyABSA or DeBERTa-based aspect-level sentiment extraction.
- **Hyperparameter Optimization**: Conduct Optuna grid search for C-regularization and n-gram boundaries.
- **Human-Annotated Test Set**: Benchmark models against manually annotated sentiment evaluation samples.
- **Cloud Deployment**: Containerize with Docker and deploy to AWS / GCP / Azure.

---

## 20. Technologies Used

- **Language & Core Libraries**: Python 3.11, Pandas, NumPy, Scikit-learn, NLTK, Joblib
- **Machine Learning Models**: Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Classifier (LinearSVC)
- **NLP & Feature Extraction**: TF-IDF Vectorizer, Custom Regular Expressions, Contraction Expansion
- **Web Frameworks**: Streamlit, Flask, Flask-CORS
- **Frontend & Visualization**: HTML5, Custom CSS3, Vanilla JavaScript, Chart.js, Plotly
