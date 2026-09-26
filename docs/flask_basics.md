# Flask & Web API Foundations: A Beginner's Guide

This guide explains how web backends work using **Flask** and how a custom JavaScript web interface communicates with Python machine learning pipelines using REST APIs.

---

## 1. What is Flask?

**Flask** is a lightweight Python web framework. It allows you to create web servers and APIs in Python using clean, simple code. 

Unlike heavy web frameworks (like Django), Flask gives you the core tools needed to build routes, handle HTTP requests, and return data to web applications.

---

## 2. What is an API?

An **API (Application Programming Interface)** acts as a messenger between two applications.

In this project:
- The **Frontend** (HTML/CSS/JavaScript) runs in your web browser.
- The **Backend** (Flask + ML Model) runs in Python on the server.

Instead of page reloads, the frontend sends structured requests to the Flask API, and Flask returns structured **JSON data**.

---

## 3. What is a Route?

A **route** is a specific URL path defined in Flask that performs an action when accessed. 

In Flask, you define routes using the `@app.route()` decorator:

```python
@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok"}
```

When a browser or client requests `http://localhost:5000/api/health`, Flask executes the `health()` function and returns the response.

---

## 4. GET vs POST HTTP Methods

HTTP requests use different methods depending on what action they perform:

| Method | Purpose | Example Use Case |
|---|---|---|
| **GET** | Retrieve data from server | Fetching analytics stats (`/api/analytics`) or checking health (`/api/health`) |
| **POST** | Send data to server to process | Sending review text to ML model (`/api/predict`) |

- **GET requests** carry data in the URL query string.
- **POST requests** send data inside the **HTTP request body** (usually as JSON).

---

## 5. What is `request.json`?

When the frontend sends a POST request with a JSON payload:

```json
{
    "review": "The taste is amazing but delivery was late."
}
```

Flask reads and parses this payload in Python using `request.get_json()` or `request.json`:

```python
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review_text = data.get("review", "")
```

---

## 6. What is `jsonify`?

`jsonify()` is a Flask helper function that converts Python dictionaries or lists into formatted JSON responses and automatically sets the HTTP header `Content-Type: application/json`.

Example:

```python
from flask import jsonify

return jsonify({
    "sentiment": "Positive",
    "confidence": 0.91
}), 200
```

---

## 7. How JavaScript `fetch()` Communicates with Flask

In modern web applications, JavaScript uses the built-in `fetch()` API to talk to backend servers asynchronously without reloading the page.

### Sending a POST request to Flask:

```javascript
// JavaScript running in browser
async function analyzeReview(reviewText) {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: reviewText })
    });

    const result = await response.json();
    console.log("Prediction from Flask:", result);
}
```

---

## 8. How the ML Model is Called from Flask

When Flask starts up, it loads the saved ML model (`models/logistic_regression_model.pkl`) and TF-IDF vectorizer (`models/tfidf_vectorizer.pkl`) into memory **ONCE**.

When a user submits a review via `/api/predict`:
1. Flask receives the review text in `request.get_json()`.
2. Flask calls `predict_review(text)` from `src/predict.py`.
3. `predict_review()` cleans text with `clean_text()`, transforms it with TF-IDF, predicts sentiment probabilities, and extracts aspect sentiments using the ABSA engine.
4. Flask wraps the result dictionary in `jsonify()` and sends it back to JavaScript.

---

## 9. Complete End-to-End JSON Data Flow

Here is how data flows step-by-step when a user analyzes a review:

```
[User types review in browser]
              ↓
[JavaScript fetch() sends POST JSON payload to /api/predict]
              ↓
[Flask receives request.json on Python server]
              ↓
[src/predict.py cleans text & runs TF-IDF vectorizer]
              ↓
[Logistic Regression model predicts overall sentiment & confidence]
              ↓
[ABSA module extracts aspect-level sentiments & clauses]
              ↓
[Flask wraps output with jsonify() and returns HTTP 200 JSON]
              ↓
[JavaScript parses response and updates HTML DOM dynamically]
              ↓
[User sees Sentiment Badge & ABSA Cards without page reload]
```
