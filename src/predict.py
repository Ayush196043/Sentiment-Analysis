import joblib
import pandas as pd
from pathlib import Path
import sys

# Import local modules cleanly
from preprocessing import clean_text
from aspect_analysis import analyze_aspect_sentiment

# Load saved deployment artifacts globally for fast inference
MODEL_DIR = Path(__file__).parent.parent / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"

print(f"Loading TF-IDF Vectorizer from: {VECTORIZER_PATH.resolve()}")
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

print(f"Loading Deployment Model (Logistic Regression) from: {MODEL_PATH.resolve()}")
sentiment_model = joblib.load(MODEL_PATH)

def predict_review(review_text: str) -> dict:
    """
    End-to-End Prediction Pipeline for New Customer Reviews.
    
    Steps:
    1. Validate input text.
    2. Apply pre-training clean_text() transformation (100% consistency).
    3. Transform text to numerical vector using pre-fitted TF-IDF vectorizer.
    4. Predict overall sentiment category & class probabilities.
    5. Run Aspect-Based Sentiment Analysis (ABSA) extraction.
    6. Return structured response dictionary.
    """
    if not review_text or not isinstance(review_text, str) or len(review_text.strip()) == 0:
        return {
            "error": "Empty or invalid review text provided.",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33},
            "aspects": []
        }
        
    # 1. Preprocess text
    cleaned = clean_text(review_text)
    
    # 2. Vectorize text
    tfidf_vec = tfidf_vectorizer.transform([cleaned])
    
    # 3. Predict sentiment category & probabilities
    prediction = sentiment_model.predict(tfidf_vec)[0]
    probs = sentiment_model.predict_proba(tfidf_vec)[0]
    
    # Map probability scores to class names
    classes = list(sentiment_model.classes_)
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
    confidence = round(float(prob_dict[prediction]), 4)
    
    # 4. Aspect-Based Sentiment Analysis
    aspects = analyze_aspect_sentiment(review_text)
    
    return {
        "original_text": review_text,
        "clean_text": cleaned,
        "sentiment": prediction,
        "confidence": confidence,
        "probabilities": prob_dict,
        "aspects": aspects
    }

if __name__ == "__main__":
    sample_review = "The taste is excellent and delicious, but the delivery was late and packaging was damaged."
    res = predict_review(sample_review)
    print("=" * 65)
    print("END-TO-END PIPELINE DEMONSTRATION")
    print("=" * 65)
    print(f"Input Review: \"{res['original_text']}\"")
    print(f"Clean Text  : \"{res['clean_text']}\"")
    print(f"Sentiment   : {res['sentiment']} (Confidence: {res['confidence']*100:.2f}%)")
    print(f"Probabilities: {res['probabilities']}")
    print("Extracted Aspects:")
    for asp in res['aspects']:
        print(f"   • {asp['aspect']:<15} -> {asp['sentiment']:<8} | Clause: \"{asp['matching_clause']}\"")
    print("=" * 65)
