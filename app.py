import os
import re
import joblib
import traceback
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# ================= FLASK INIT =================
app = Flask(__name__)
CORS(app)

# ================= NLTK SETUP =================
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ================= HELPERS =================
def sanitize_text(text):
    """Remove HTML tags and strip text."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.strip()

def preprocess_text(text):
    """Lowercase, remove non-alphabetic, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def is_url(text):
    """Detect if input is URL."""
    return bool(re.match(r'^(http|https)://', text.strip()))

def extract_text_from_url(url):
    """Ka soo saar qoraalka bogga webka URL"""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ")
        return text.strip()
    except Exception:
        return ""

# ================= LOAD MODELS =================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVED_MODEL_DIR = os.path.join(BASE_DIR, "..", "saved_model")

    MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "svm_high_confidence.pkl")
    VECTORIZER_PATH = os.path.join(SAVED_MODEL_DIR, "fake_real_TF_IDF_vectorizer.pkl")
    ENCODER_PATH = os.path.join(SAVED_MODEL_DIR, "fake_real_label_encoder.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    print("[SUCCESS] Models loaded successfully")

except Exception as e:
    print("[ERROR] Loading models failed:", e)
    traceback.print_exc()
    exit(1)

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "Fake News Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON lama helin"}), 400

        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error": "Qoraal lama soo dirin"}), 400

        content = str(content).strip()

        # Haddii input uu URL yahay → ka soo saar text
        if is_url(content):
            content = extract_text_from_url(content)
            if not content:
                return jsonify({"error": "Qoraalka laga helay URL-ka lama heli karo"}), 400

        # ================= Preprocess =================
        clean_input = preprocess_text(content)

        # Vectorize & Predict
        X = vectorizer.transform([clean_input])
        expected_features = model.coef_.shape[1]

        if X.shape[1] != expected_features:
            diff = expected_features - X.shape[1]
            if diff > 0:
                X = np.hstack([X.toarray(), np.zeros((X.shape[0], diff))])
            else:
                X = X.toarray()[:, :expected_features]
        else:
            X = X.toarray()

        pred = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            confidence = round(float(max(probs)) * 100, 2)
        else:
            score = model.decision_function(X)
            score = abs(score[0])
            confidence = round((1 / (1 + np.exp(-score))) * 100, 2)

        label = label_encoder.inverse_transform([pred])[0]
        result = "REAL NEWS" if label == 1 else "FAKE NEWS"

        return jsonify({"prediction": result, "confidence": f"{confidence}%"})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error ayaa dhacay"}), 500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Data lama helin"}), 400

        name = data.get("name")
        email = data.get("email")
        message = data.get("message")

        if not all([name, email, message]):
            return jsonify({"error": "Fadlan buuxi dhamaan meelaha banaan"}), 400

        # Log to file
        with open("contacts.txt", "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")

        print(f"[*] New message from {name} ({email})")
        return jsonify({"status": "Success", "message": "Fariintaada waa nala soo gaarsiiyey!"})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error ayaa dhacay"}), 500

# ================= RUN SERVER =================
if __name__ == "__main__":
    print("[*] Flask server starting...")
    app.run(host="0.0.0.0", port=3402, debug=False)
