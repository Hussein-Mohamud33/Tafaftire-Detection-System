import os
import re
import joblib
import traceback
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ================= FLASK INIT =================
app = Flask(__name__)
CORS(app)

# ================= NLTK SETUP =================
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ================= HELPERS =================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    # Limit text length (MUHIIM SPEED)
    text = text[:1500]

    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()

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
        resp = requests.get(url, timeout=1.5)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator=" ").strip()

        # Limit URL content
        return text[:2000]

    except:
        return ""

# ================= LOAD MODELS =================
model, vectorizer, label_encoder = None, None, None
BASE_DIR = os.getcwd()
MODEL_FOLDER = os.path.join(BASE_DIR, "saved_model")

def load_models():
    global model, vectorizer, label_encoder
    try:
        MODEL_PATH = os.path.join(MODEL_FOLDER, "svm_high_confidence.pkl")
        VECTORIZER_PATH = os.path.join(MODEL_FOLDER, "fake_real_TF_IDF_vectorizer.pkl")
        ENCODER_PATH = os.path.join(MODEL_FOLDER, "fake_real_label_encoder.pkl")

        if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]):
            print("❌ Model files not found.")
            return

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)

        print("✅ Models loaded successfully")

    except Exception as e:
        print("❌ Failed to load models:", e)
        traceback.print_exc()

load_models()

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "Fake News Detection API running"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON data not found"}), 400

        input_type = data.get("type", "text")
        content = str(data.get("data", "")).strip()

        if not content:
            return jsonify({"error": "No text or URL provided"}), 400

        # URL Mode
        if input_type == "url":
            if not is_url(content):
                return jsonify({"error": "Invalid URL"}), 400
            content = extract_text_from_url(content)
            if not content:
                return jsonify({"error": "Cannot extract text"}), 400

        clean_input = preprocess_text(content)

        if not clean_input:
            return jsonify({"error": "Processed text empty"}), 400

        # VECTORIZE (NO toarray → FAST)
        X = vectorizer.transform([clean_input])

        # PREDICT DIRECTLY (Sparse OK)
        pred = model.predict(X)[0]

        # CONFIDENCE
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            confidence = round(float(max(probs)) * 100, 2)
        else:
            score = model.decision_function(X)
            score = abs(score[0])
            confidence = round((1 / (1 + np.exp(-score))) * 100, 2)

        label = label_encoder.inverse_transform([pred])[0]
        result = "REAL NEWS" if label == 1 else "FAKE NEWS"

        return jsonify({
            "prediction": result,
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error occurred: {str(e)}"}), 500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Data not found"}), 400

        name = data.get("name")
        email = data.get("email")
        message = data.get("message")

        if not all([name, email, message]):
            return jsonify({"error": "Please fill all fields"}), 400

        with open("contacts.txt", "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")

        return jsonify({"status": "Success", "message": "Message received!"})

    except:
        traceback.print_exc()
        return jsonify({"error": "Server error occurred"}), 500

# ================= RUN SERVER =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3402))
    print(f"[*] Flask server starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
