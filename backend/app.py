import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

# Setup project pathing
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "results"
FRONTEND_DIR = ROOT_DIR / "frontend"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Initialize Flask App with static folder pointing to frontend
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)  # Enable Cross-Origin Resource Sharing for dev flexibility

# Import prediction pipeline safely
try:
    from predict import predict_review, tfidf_vectorizer, sentiment_model
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR = str(e)
    print(f"[ERROR] Failed to load prediction pipeline: {e}")

# Helper to load analytics data
def get_analytics_data():
    aspect_csv = RESULTS_DIR / "aspect_summary.csv"
    model_csv = RESULTS_DIR / "model_comparison.csv"
    
    aspect_summary = []
    model_comparison = []
    
    if aspect_csv.exists():
        df_aspect = pd.read_csv(aspect_csv)
        aspect_summary = df_aspect.to_dict(orient="records")
        
    if model_csv.exists():
        df_model = pd.read_csv(model_csv)
        model_comparison = df_model.to_dict(orient="records")
        
    return {
        "kpis": {
            "total_reviews": 393579,
            "positive_pct": 77.94,
            "neutral_pct": 7.56,
            "negative_pct": 14.50
        },
        "rating_distribution": [
            {"rating": "5 Stars", "count": 248773, "pct": 63.21},
            {"rating": "4 Stars", "count": 33006, "pct": 8.39},
            {"rating": "3 Stars", "count": 29769, "pct": 7.56},
            {"rating": "2 Stars", "count": 29763, "pct": 7.56},
            {"rating": "1 Star", "count": 52268, "pct": 13.28}
        ],
        "aspect_summary": aspect_summary,
        "model_comparison": model_comparison
    }

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/")
def serve_index():
    """Serve main frontend SPA index.html"""
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend assets (css, js, images)"""
    return send_from_directory(str(FRONTEND_DIR), path)

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server and ML model availability."""
    if MODEL_LOADED and tfidf_vectorizer is not None and sentiment_model is not None:
        return jsonify({
            "status": "ok",
            "model": "Logistic Regression",
            "vectorizer": "TF-IDF (ngram_range=(1,2), max_features=20000)",
            "macro_f1": "67.04%"
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": f"ML Model pipeline failed to load: {MODEL_ERROR if 'MODEL_ERROR' in globals() else 'Unknown error'}"
        }), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint. Accepts JSON {"review": "..."}, calls src/predict.py,
    and returns structured prediction with overall sentiment and ABSA results.
    """
    if not MODEL_LOADED:
        return jsonify({
            "error": "ML Prediction Pipeline is unavailable on server."
        }), 500

    data = request.get_json(silent=True)
    if not data or "review" not in data:
        return jsonify({
            "error": "Invalid request. Missing 'review' key in JSON payload."
        }), 400

    review_text = data.get("review", "")
    if not isinstance(review_text, str) or not review_text.strip():
        return jsonify({
            "error": "Review text cannot be empty or non-string."
        }), 400

    if len(review_text) > 5000:
        return jsonify({
            "error": "Review text is too long (maximum 5,000 characters allowed)."
        }), 400

    try:
        result = predict_review(review_text)
        return jsonify({
            "status": "success",
            "data": result,
            "disclaimer": "Predictions are model-generated probability estimates and may contain errors."
        }), 200
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return jsonify({
            "error": "An error occurred while processing the review."
        }), 500

@app.route("/api/analytics", methods=["GET"])
def analytics():
    """Returns dataset analytics and aspect summary statistics."""
    try:
        data = get_analytics_data()
        return jsonify({
            "status": "success",
            "data": data
        }), 200
    except Exception as e:
        print(f"[ERROR] Analytics fetching failed: {e}")
        return jsonify({
            "error": "Failed to fetch analytics dataset."
        }), 500

if __name__ == "__main__":
    print("==================================================")
    print("Starting Flask AI Customer Feedback API Server...")
    print("Serving frontend from:", FRONTEND_DIR)
    print("URL: http://127.0.0.1:5000")
    print("==================================================")
    app.run(host="127.0.0.1", port=5000, debug=True)
