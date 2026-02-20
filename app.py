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
CORS(app, origins=["https://tafaftire.netlify.app"])

# ================= NLTK SETUP =================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ================= HELPERS =================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

def is_url(text):
    return bool(re.match(r'^(http|https)://', text.strip()))

def extract_text_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator=" ").strip()
    except Exception:
        return ""

# ================= LOAD MODELS =================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")
    ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    print("✅ Models loaded")

except Exception as e:
    print("❌ Model loading failed:", e)
    traceback.print_exc()
    exit(1)

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Backend Running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Qoraal lama soo dirin"}), 400

        content = str(data["text"]).strip()

        if is_url(content):
            content = extract_text_from_url(content)
            if not content:
                return jsonify({"error": "URL qoraal laga heli waayay"}), 400

        clean_input = preprocess_text(content)
        X = vectorizer.transform([clean_input]).toarray()

        pred = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            confidence = round(float(max(model.predict_proba(X)[0])) * 100, 2)
        else:
            score = abs(model.decision_function(X)[0])
            confidence = round((1 / (1 + np.exp(-score))) * 100, 2)

        label = label_encoder.inverse_transform([pred])[0]

        result = "REAL NEWS" if str(label).lower() == "real" else "FAKE NEWS"

        return jsonify({
            "prediction": result,
            "confidence": f"{confidence}%"
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error ayaa dhacay"}), 500


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3402))
    app.run(host="0.0.0.0", port=port)
