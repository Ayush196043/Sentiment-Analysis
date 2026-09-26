import json
from pathlib import Path

def build_preprocessing_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 4 — NLP Text Preprocessing Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Clean and normalize 393,579 customer reviews into NLP-ready text while preserving negation context.\n",
                    "\n",
                    "### Workflow:\n",
                    "1. Load `data/processed/reviews_deduplicated.csv`\n",
                    "2. Apply `clean_text()` (Lowercasing, Contraction Expansion, HTML & URL cleanup, Symbol removal)\n",
                    "3. Demonstrate Negation Preservation\n",
                    "4. Demonstrate Tokenization\n",
                    "5. Conduct Stopword Handling Experiment\n",
                    "6. Conduct Stemming vs Lemmatization Experiment\n",
                    "7. Inspect 10 Real Dataset Before/After Examples\n",
                    "8. Quality Checks & Save to `data/processed/reviews_cleaned.csv`"
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
                    "import re\n",
                    "from pathlib import Path\n",
                    "import nltk\n",
                    "from nltk.corpus import stopwords\n",
                    "from nltk.stem import PorterStemmer, WordNetLemmatizer\n",
                    "\n",
                    "# Import reusable clean_text module from src\n",
                    "import sys\n",
                    "sys.path.append('../src')\n",
                    "from preprocessing import clean_text, expand_contractions, NEGATION_WORDS\n",
                    "\n",
                    "print('Libraries imported successfully.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Verified Deduplicated Dataset\n",
                    "We load `reviews_deduplicated.csv` (393,579 rows) generated in Phase 3."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = Path('../data/processed/reviews_deduplicated.csv')\n",
                    "df = pd.read_csv(data_path)\n",
                    "print(f'Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns')\n",
                    "display(df[['Id', 'Score', 'Sentiment', 'Text']].head(3))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Text Preprocessing Function & Cleaning Pipeline\n",
                    "We apply lowercasing, contraction expansion (`isn't` -> `is not`), HTML cleanup (`<br />` -> space), URL cleanup, and punctuation removal."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df['clean_text'] = df['Text'].astype(str).apply(clean_text)\n",
                    "print('Cleaned text column generated.')\n",
                    "display(df[['Text', 'clean_text']].head(3))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Negation Preservation Verification\n",
                    "Negation words (`not`, `no`, `never`, `cannot`, `n't`) reverse sentiment polarity. We verify that phrases like `not good` are NOT corrupted into `good`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "test_negations = [\n",
                    "    'The food was not good at all.',\n",
                    "    'I am never buying this brand again.',\n",
                    "    'This product is not worth the price!',\n",
                    "    'It wasn\\'t great and tasted terrible.'\n",
                    "]\n",
                    "for t in test_negations:\n",
                    "    print(f'BEFORE: {t}')\n",
                    "    print(f'AFTER : {clean_text(t)}\\n')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Tokenization Demonstration\n",
                    "Splitting clean text into individual word tokens for feature extraction."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sample = df['clean_text'].iloc[0]\n",
                    "tokens = sample.split()\n",
                    "print('Sample Clean Text:', sample)\n",
                    "print('Tokens:', tokens[:15])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Stopword Handling Experiment\n",
                    "Standard NLTK stopword removal deletes `not` and `no`. We demonstrate why a custom negation-preserving stopword set is mandatory."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "nltk_stops = set(stopwords.words('english'))\n",
                    "custom_stops = nltk_stops - NEGATION_WORDS\n",
                    "\n",
                    "phrase = 'This coffee is not good and I do not like it.'\n",
                    "print('Original Phrase:', phrase)\n",
                    "print('Standard NLTK Stopwords (DESTROYS NEGATION!):', ' '.join([w for w in clean_text(phrase).split() if w not in nltk_stops]))\n",
                    "print('Custom Negation-Preserving Stopwords:        ', ' '.join([w for w in clean_text(phrase).split() if w not in custom_stops]))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Stemming vs. Lemmatization Experiment\n",
                    "Comparing PorterStemmer vs. WordNetLemmatizer on food domain terms."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "stemmer = PorterStemmer()\n",
                    "lemmatizer = WordNetLemmatizer()\n",
                    "words = ['canned', 'running', 'better', 'quality', 'tasted', 'goods']\n",
                    "\n",
                    "for w in words:\n",
                    "    print(f'{w:<10} | Stemmed: {stemmer.stem(w):<10} | Lemmatized: {lemmatizer.lemmatize(w)}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Quality Checks & Export\n",
                    "Verify zero empty clean reviews and export to `data/processed/reviews_cleaned.csv`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "empty_clean = (df['clean_text'].apply(lambda s: len(s.split())) == 0).sum()\n",
                    "print(f'Empty Cleaned Text Count: {empty_clean}')\n",
                    "print(f'Total Rows Preserved: {len(df):,}')\n",
                    "\n",
                    "output_path = Path('../data/processed/reviews_cleaned.csv')\n",
                    "df[['Id', 'ProductId', 'UserId', 'Score', 'Sentiment', 'Text', 'clean_text']].to_csv(output_path, index=False)\n",
                    "print(f'Cleaned dataset saved to: {output_path.resolve()}')"
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
    
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "02_preprocessing.ipynb"
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    print(f"Created notebook: {nb_path.resolve()}")

def build_eda_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 3 — Exploratory Data Analysis (EDA) Notebook\n",
                    "\n",
                    "**Project:** AI-Powered Customer Feedback Intelligence System  \n",
                    "**Goal:** Analyze data quality, score distribution, sentiment distribution, and review length statistics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from pathlib import Path\n",
                    "\n",
                    "df = pd.read_csv('../data/processed/reviews_with_sentiment.csv')\n",
                    "print('Processed Data Loaded:', df.shape)\n",
                    "display(df.head(3))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Duplicate Analysis & Deduplication\n",
                    "Removing 174,875 duplicate review texts to prevent train/test data leakage."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_dedup = df.drop_duplicates(subset=['Text']).copy()\n",
                    "print(f'Deduplicated Rows: {len(df_dedup):,}')\n",
                    "print(df_dedup['Sentiment'].value_counts(normalize=True)*100)"
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
    
    nb_path = Path("notebooks/01_eda.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {nb_path.resolve()}")

if __name__ == "__main__":
    build_preprocessing_notebook()
    build_eda_notebook()
