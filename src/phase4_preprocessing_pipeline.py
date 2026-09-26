import pandas as pd
import numpy as np
import re
from pathlib import Path
from preprocessing import clean_text, NEGATION_WORDS
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def run_phase4_pipeline():
    input_path = Path("data/processed/reviews_deduplicated.csv")
    output_path = Path("data/processed/reviews_cleaned.csv")
    
    print("=" * 75)
    print("PHASE 4 — NLP TEXT PREPROCESSING PIPELINE")
    print("=" * 75)
    
    # ---------------------------------------------------------
    # STEP 1: LOAD DEDUPLICATED DATASET
    # ---------------------------------------------------------
    print("\n--- STEP 1: LOADING DEDUPLICATED DATASET ---")
    df = pd.read_csv(input_path)
    print(f"Dataset Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Text values: {df['Text'].isnull().sum()}")
    print(f"Missing Sentiment values: {df['Sentiment'].isnull().sum()}")
    
    # ---------------------------------------------------------
    # STEP 2 & 3 & 4 & 5 & 6 & 7: APPLYING CLEAN_TEXT FUNCTION
    # ---------------------------------------------------------
    print("\n--- STEP 2-7: APPLYING REUSABLE TEXT CLEANING ---")
    print("Applying clean_text() to all reviews (Lowercasing, HTML removal, URL removal, Contraction expansion, Symbol cleanup, Negation preservation)...")
    
    df['clean_text'] = df['Text'].astype(str).apply(clean_text)
    print("Text cleaning completed successfully!")
    
    # ---------------------------------------------------------
    # STEP 7: DEMONSTRATING NEGATION PRESERVATION
    # ---------------------------------------------------------
    print("\n--- STEP 7: DEMONSTRATING NEGATION PRESERVATION ---")
    negation_samples = [
        "The food was not good at all.",
        "I am never buying this brand again.",
        "This product is not worth the price!",
        "It wasn't great and tasted terrible.",
        "There is no way I can recommend this."
    ]
    print(f"{'Original Text':<45} | {'Cleaned Text (Negation Preserved)'}")
    print("-" * 90)
    for sample in negation_samples:
        print(f"{sample:<45} | {clean_text(sample)}")
        
    # ---------------------------------------------------------
    # STEP 8: TOKENIZATION DEMONSTRATION
    # ---------------------------------------------------------
    print("\n--- STEP 8: TOKENIZATION DEMONSTRATION ---")
    sample_review = df['clean_text'].iloc[0]
    tokens = sample_review.split()
    print("Sample Cleaned Text:", sample_review)
    print(f"Tokenization Output (First 15 tokens out of {len(tokens)}):", tokens[:15])
    
    # ---------------------------------------------------------
    # STEP 9: STOPWORD HANDLING EXPERIMENT
    # ---------------------------------------------------------
    print("\n--- STEP 9: STOPWORD HANDLING EXPERIMENT ---")
    default_nltk_stopwords = set(stopwords.words('english'))
    custom_negation_stopwords = default_nltk_stopwords - NEGATION_WORDS
    
    test_phrase = "This coffee is not good and I don't like it."
    print("Test Phrase:", test_phrase)
    
    # Standard NLTK stopword removal (Destroys Negation!)
    tokens_standard = [w for w in clean_text(test_phrase).split() if w not in default_nltk_stopwords]
    print("Standard Stopword Removal (DESTROYS NEGATION!): ", " ".join(tokens_standard))
    
    # Custom Negation-Preserving stopword removal
    tokens_custom = [w for w in clean_text(test_phrase).split() if w not in custom_negation_stopwords]
    print("Custom Negation-Preserving Stopword Removal:     ", " ".join(tokens_custom))
    
    # ---------------------------------------------------------
    # STEP 10: STEMMING vs LEMMATIZATION DEMONSTRATION
    # ---------------------------------------------------------
    print("\n--- STEP 10: STEMMING vs LEMMATIZATION EXPERIMENT ---")
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    demo_words = ["canned", "running", "better", "quality", "tasted", "badly", "goods", "studies"]
    print(f"{'Original Word':<15} | {'Stemmed (Porter)':<18} | {'Lemmatized (WordNet)'}")
    print("-" * 55)
    for w in demo_words:
        print(f"{w:<15} | {stemmer.stem(w):<18} | {lemmatizer.lemmatize(w, pos='v') if w in ['running', 'tasted'] else lemmatizer.lemmatize(w)}")
        
    # ---------------------------------------------------------
    # STEP 11: BEFORE / AFTER ANALYSIS TABLE (10+ Examples)
    # ---------------------------------------------------------
    print("\n--- STEP 11: BEFORE / AFTER ANALYSIS TABLE (10 Real Dataset Examples) ---")
    # Find diverse examples containing HTML, URLs, UPPERCASE, Contractions, Negations
    indices_to_show = []
    
    for idx, row in df.iterrows():
        t = str(row['Text'])
        if '<br' in t and len(indices_to_show) < 2:
            indices_to_show.append(idx)
        elif ('http' in t or 'www' in t) and len(indices_to_show) < 4:
            indices_to_show.append(idx)
        elif ("n't" in t or "not" in t) and len(indices_to_show) < 7:
            indices_to_show.append(idx)
        elif len(indices_to_show) < 10 and idx not in indices_to_show:
            indices_to_show.append(idx)
        if len(indices_to_show) >= 10:
            break
            
    samples_df = df.iloc[indices_to_show][['Score', 'Sentiment', 'Text', 'clean_text']].copy()
    for i, (_, row) in enumerate(samples_df.iterrows(), 1):
        orig_snippet = row['Text'][:100].replace('\n', ' ') + ("..." if len(row['Text']) > 100 else "")
        clean_snippet = row['clean_text'][:100] + ("..." if len(row['clean_text']) > 100 else "")
        print(f"\n[Example {i}] Score {row['Score']} ({row['Sentiment']}):")
        print(f"  BEFORE: \"{orig_snippet}\"")
        print(f"  AFTER : \"{clean_snippet}\"")
        
    # ---------------------------------------------------------
    # STEP 12: QUALITY CHECK ON CLEANED TEXT
    # ---------------------------------------------------------
    print("\n--- STEP 12: QUALITY CHECK ON CLEANED TEXT ---")
    df['clean_char_count'] = df['clean_text'].apply(len)
    df['clean_word_count'] = df['clean_text'].apply(lambda s: len(s.split()))
    
    empty_clean = (df['clean_word_count'] == 0).sum()
    very_short_clean = (df['clean_word_count'] < 3).sum()
    
    print(f"Original Dataset Rows: {len(df):,}")
    print(f"Rows with Empty Cleaned Text (0 words): {empty_clean:,} ({empty_clean/len(df)*100:.4f}%)")
    print(f"Rows with Very Short Cleaned Text (< 3 words): {very_short_clean:,} ({very_short_clean/len(df)*100:.4f}%)")
    print(f"Average Words Before Preprocessing: {df['Text'].apply(lambda s: len(str(s).split())).mean():.2f}")
    print(f"Average Words After Preprocessing : {df['clean_word_count'].mean():.2f}")
    print("-> Decision: 0 rows will be deleted. All 393,579 rows preserved.")
    
    # ---------------------------------------------------------
    # STEP 13: SAVE PROCESSED DATA
    # ---------------------------------------------------------
    print("\n--- STEP 13: SAVING CLEANED DATASET ---")
    output_cols = ['Id', 'ProductId', 'UserId', 'Score', 'Sentiment', 'Text', 'clean_text']
    # Select available columns safely
    actual_cols = [c for c in output_cols if c in df.columns]
    
    df[actual_cols].to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    run_phase4_pipeline()
