# Technical Interview Questions & Answers: Customer Feedback Intelligence

This document contains 30 comprehensive technical interview questions, project-specific empirical metric Q&As, and common trap questions for interviewing on this project.

---

## Part 1: Core ML & NLP Technical Questions (1–30)

### 1. Why did you choose the Amazon Fine Food Reviews dataset?
**Answer**: It is a large, real-world customer review corpus containing over 568,000 reviews with rich text and star ratings. It provides a realistic environment featuring severe class imbalance, noise, customer colloquialisms, and multi-aspect feedback.

### 2. Why did you use TF-IDF vectorization?
**Answer**: TF-IDF (Term Frequency-Inverse Document Frequency) efficiently converts unstructured text into numerical features by balancing word frequency against document-wide rarity. It penalizes non-informative words while highlighting strong sentiment signals.

### 3. What mathematically is TF-IDF?
**Answer**: TF-IDF is the product of Term Frequency ($\text{TF}$) and Inverse Document Frequency ($\text{IDF}$):
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$
where $\text{TF}(t, d)$ is the count of term $t$ in document $d$, $N$ is the total document count in corpus $D$, and $\text{DF}(t)$ is the number of documents containing term $t$.

### 4. Why did you use n-grams (1, 2)?
**Answer**: Unigrams ($1$-grams) capture single words ('good', 'bad'), but lose negation context. Bigrams ($2$-grams) preserve phrase patterns like 'not good', 'very bad', or 'highly recommend', significantly improving sentiment precision.

### 5. Why set `max_features=20000`?
**Answer**: Unconstrained TF-IDF creates a vocabulary exceeding 200,000 unique terms, mostly noisy typos or rare names. Limiting to the top 20,000 most informative features reduces computational memory, speeds up training, and prevents overfitting.

### 6. Why perform a Train/Test split?
**Answer**: To evaluate how well the trained machine learning model generalizes to completely unseen customer reviews, avoiding memorization (overfitting).

### 7. Why use stratified sampling for the split?
**Answer**: The dataset has an imbalanced class distribution (77.94% Positive, 14.50% Negative, 7.56% Neutral). Stratified sampling ensures that both train (80%) and test (20%) sets maintain identical class proportions.

### 8. What is data leakage, and how did you prevent it?
**Answer**: Data leakage occurs when information from the test set leaks into the training pipeline. I prevented leakage by fitting `TfidfVectorizer` **strictly** on `X_train` and applying `transform()` to `X_test`. Additionally, removing duplicate reviews before splitting prevented identical review text from appearing in both train and test splits.

### 9. Why compare Naive Bayes, Logistic Regression, and Linear SVM?
**Answer**: They represent three distinct classical baseline families:
- **Multinomial Naive Bayes**: Generative probabilistic model.
- **Logistic Regression**: Discriminative linear probability model.
- **Linear SVM**: Maximum-margin hyperplane decision boundary.
Comparing them allowed me to select the best architecture based on empirical evidence rather than assumptions.

### 10. Why wasn't Accuracy sufficient as an evaluation metric?
**Answer**: Due to class imbalance (77.9% Positive), a dummy model predicting "Positive" for every sample would achieve 77.9% accuracy while completely failing on Negative and Neutral reviews.

### 11. What is Macro F1-Score, and why did you select it?
**Answer**: Macro F1 computes the unweighted mean of F1-scores across all classes:
$$\text{Macro F1} = \frac{\text{F1}_{\text{Positive}} + \text{F1}_{\text{Neutral}} + \text{F1}_{\text{Negative}}}{3}$$
It treats all classes equally, penalizing models that perform poorly on minority classes like Neutral or Negative.

### 12. Why was the Neutral class (3 stars) difficult to predict?
**Answer**: 3-star reviews are inherently ambiguous. Customers often write mixed feedback (e.g., "Good product but bad shipping"), using positive vocabulary for product quality and negative vocabulary for logistics.

### 13. What did Error Analysis reveal?
**Answer**: Inspecting 9,348 test errors showed:
1. 76.7% of errors involved the Neutral class.
2. Misclassified reviews were 21.8% longer on average, containing conflicting sentiments.
3. Rating-derived proxy label mismatches (polite text paired with 1-star ratings).

### 14. What are rating-derived proxy labels?
**Answer**: Target labels created by mapping numerical star ratings (1–2 $\to$ Negative, 3 $\to$ Neutral, 4–5 $\to$ Positive). They serve as practical proxies but contain noise compared to human-annotated sentiment text.

### 15. What is Aspect-Based Sentiment Analysis (ABSA)?
**Answer**: ABSA decomposes text into sub-topics, extracting individual sentiment polarity for specific product aspects (e.g., Taste vs. Delivery) within a single review.

### 16. Why is overall sentiment classification insufficient for businesses?
**Answer**: An overall "Negative" classification tells a business a customer is unhappy, but does not indicate *why*. ABSA identifies whether the failure was product quality, pricing, packaging damage, or delivery speed.

### 17. How does your baseline ABSA module work?
**Answer**: It splits reviews into clauses, scans clauses for domain keyword dictionaries across 7 food categories (`Taste`, `Packaging`, `Price`, `Ingredients`, `Delivery`, `Quality`, `Customer Support`), and assigns clause-level sentiment polarity.

### 18. What are the limitations of keyword-based ABSA?
**Answer**: It relies on pre-defined keyword lists and cannot handle implicit sentiment, complex syntactic dependencies, or novel vocabulary as effectively as Transformer models.

### 19. Why was Logistic Regression selected as the deployment model?
**Answer**: It achieved the highest Macro F1 (**67.04%**) and Overall Accuracy (**88.12%**), while providing calibrated probability scores (`predict_proba()`) required for model confidence displays.

### 20. Why save both the model AND the vectorizer?
**Answer**: The trained Logistic Regression model expects numerical input vectors with the exact 20,000 feature mapping created during training. New raw text must be transformed using the exact same `tfidf_vectorizer.pkl`.

### 21. Why must preprocessing remain 100% consistent between training and inference?
**Answer**: Any mismatch (e.g., stripping negation during inference when it was preserved during training) alters token vocabulary, corrupting feature vectors and causing invalid predictions.

### 22. What happens step-by-step when a new review arrives?
**Answer**: `predict_review()` accepts raw text $\to$ applies `clean_text()` $\to$ transforms with `tfidf_vectorizer.pkl` $\to$ computes class probabilities with `logistic_regression_model.pkl` $\to$ extracts aspects via `analyze_aspect_sentiment()` $\to$ returns structured JSON.

### 23. How does Flask communicate with JavaScript?
**Answer**: The browser sends an asynchronous HTTP POST request (`fetch('/api/predict', { body: JSON.stringify({review}) })`). Flask parses `request.json`, runs inference, and returns JSON formatted with `jsonify()`.

### 24. What is the difference between GET and POST in your API?
**Answer**:
- **GET**: Used for idempotent data retrieval (`/api/health` and `/api/analytics`).
- **POST**: Used to send review text payloads in the HTTP request body for server processing (`/api/predict`).

### 25. What is JSON?
**Answer**: JavaScript Object Notation—a lightweight, human-readable data format used to transmit structured data between backend web servers and frontend interfaces.

### 26. Why use Flask for the backend?
**Answer**: Flask is a lightweight Python web micro-framework ideal for wrapping ML models into REST APIs without unnecessary administrative overhead.

### 27. Why keep the Streamlit application alongside the Flask web app?
**Answer**: Streamlit provides a fast, Python-native dashboard for exploratory analytics, while the custom HTML/CSS/JS + Flask stack provides a decoupled production-ready microservice architecture.

### 28. What would you improve with more engineering time?
**Answer**: Implement fine-tuned Transformer models (BERT/DeBERTa), deploy PyABSA for neural aspect extraction, build a human-annotated test benchmark, and deploy via Docker on cloud infrastructure.

### 29. How would you use BERT/Transformers for this task?
**Answer**: Fine-tune a pre-trained `bert-base-uncased` sequence classification model on `X_train` using Hugging Face `Transformers` and PyTorch, leveraging attention mechanisms to capture contextual sentiment and negation.

### 30. How would you deploy this system in production?
**Answer**: Wrap the Flask API in a Docker container, deploy to AWS ECS/EKS behind an Application Load Balancer, store model artifacts on AWS S3, and set up Prometheus/Grafana to monitor prediction latency and data drift.

---

## Part 2: Empirical Dataset & Project-Specific Questions

- **Dataset Size**: 568,454 original raw reviews $\to$ 174,875 duplicates removed $\to$ **393,579 clean unique reviews**.
- **Class Breakdown**: Positive: **306,779** (77.94%), Negative: **57,031** (14.50%), Neutral: **29,769** (7.56%).
- **Model Results**:
  - Multinomial Naive Bayes: Accuracy = 84.81%, Macro F1 = 56.05%
  - Logistic Regression: Accuracy = **88.12%**, Macro F1 = **67.04%** (Selected)
  - Linear SVM: Accuracy = 88.08%, Macro F1 = 66.60%
- **ABSA Volume**: Processed 78,716 test reviews, extracting **133,911 aspect mentions** across 7 food domain topics.

---

## Part 3: Common Trap Questions & Bulletproof Responses

### Q: "If your accuracy is 88.12%, why isn't your model considered near-perfect?"
**Answer**: "Because 77.9% of the dataset is Positive. A baseline predicting 'Positive' for everything reaches ~78% accuracy. The true metric is Macro F1 (67.04%), which reveals that Neutral review classification remains challenging (Neutral F1 is 32.06%)."

### Q: "Why not simply use star ratings directly instead of building an ML model?"
**Answer**: "Star ratings exist for historical reviews, but new unrated feedback (e.g., customer support emails, chat transcripts, social media comments) has no star rating. ML allows us to predict sentiment on unrated text."

### Q: "Why didn't you start with BERT or LLMs immediately?"
**Answer**: "Good ML practice starts with establishing baseline models (Naive Bayes, Logistic Regression) using classic feature engineering (TF-IDF). Baseline models are fast, explainable, cheap to inference, and provide a benchmark to justify the compute cost of neural networks."

### Q: "Does model probability equal guaranteed correctness?"
**Answer**: "No. Logistic Regression probability outputs reflect proximity to the decision boundary in feature space, not guaranteed truth. That is why our UI includes explicit disclaimers for end users."
