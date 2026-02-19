import os
import re
import joblib
import traceback
import numpy as np
import requests
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# ================= FLASK INIT =================
app = Flask(__name__, static_folder='Front_End_Data', static_url_path='')
CORS(app, resources={r"/*": {"origins": ["https://tafaftire.netlify.app", "http://localhost:3402", "http://127.0.0.1:3402"]}})

# ================= NLTK SETUP =================
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ================= HELPERS =================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
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
    SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

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
    return app.send_static_file('Index.html')

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

        # Vectorize
        X = vectorizer.transform([clean_input])

        # ===== FEATURE MISMATCH FIX =====
        expected_features = model.coef_.shape[1]

        if X.shape[1] != expected_features:
            diff = expected_features - X.shape[1]

            if diff > 0:
                X = np.hstack([X.toarray(), np.zeros((X.shape[0], diff))])
            else:
                X = X.toarray()[:, :expected_features]
        else:
            X = X.toarray()
        # =================================

        pred = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            confidence = round(float(max(model.predict_proba(X)[0])) * 100, 2)
        else:
            score = abs(model.decision_function(X)[0])
            confidence = round((1 / (1 + np.exp(-score))) * 100, 2)

        label = label_encoder.inverse_transform([pred])[0]
        result = "REAL NEWS" if label == 1 else "FAKE NEWS"

        return jsonify({
            "prediction": result,
            "confidence": f"{confidence}%"
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error ayaa dhacay"}), 500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data = request.get_json()
        name = data.get("name")
        email = data.get("email")
        message = data.get("message")

        if not all([name, email, message]):
            return jsonify({"error": "Fadlan buuxi meelaha banaan"}), 400

        with open("contacts.txt", "a", encoding="utf-8") as f:
            f.write(f"{name} | {email} | {message}\n")

        return jsonify({"status": "Success", "message": "Fariinta waa la helay!"})

    except Exception:
        return jsonify({"error": "Server error"}), 500

# ================= RUN =================
if __name__ == "__main__":
    # Render wuxuu bixiyaa port gaar ah oo lagu magacaabo PORT
    port = int(os.environ.get("PORT", 3402))
    
    # Maadaama uu hadda Online yahay, uma baahna webbrowser.open
    app.run(host="0.0.0.0", port=port, debug=False)
