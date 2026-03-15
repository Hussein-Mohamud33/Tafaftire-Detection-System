import os
import re
import json
import time
import random
import sys

# ================= WINDOWS UTF FIX =================
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ================= FLASK =================
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="Front_End", static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = os.environ.get("SECRET_KEY", "tafaftire-key")

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= DATA DIR =================
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")

os.makedirs(DATA_DIR, exist_ok=True)

STATS_FILE = os.path.join(DATA_DIR, "stats.json")
ANALYSIS_HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.json")

# ================= MODEL GLOBALS =================
_model = None
_vectorizer = None
_label_encoder = None

# ================= LOAD MODEL =================
def get_ml_resources():

    global _model, _vectorizer, _label_encoder

    if _model is not None:
        return _model, _vectorizer, _label_encoder

    try:

        import joblib

        model_path = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
        vec_path = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")
        enc_path = os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl")

        if os.path.exists(model_path):

            _model = joblib.load(model_path)
            _vectorizer = joblib.load(vec_path)
            _label_encoder = joblib.load(enc_path)

            print("[OK] Model Loaded")

        else:

            print("[WARNING] No ML model found. Using heuristic mode.")

    except Exception as e:

        print("Model loading error:", e)

    return _model, _vectorizer, _label_encoder


# ================= STATS =================
def load_stats():

    if os.path.exists(STATS_FILE):

        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)

        except:
            pass

    return {"requests_handled": 0}


def save_stats(stats):

    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except:
        pass


global_stats = load_stats()

# ================= NLP =================
_nlp_tools = None

def get_nlp_tools():

    global _nlp_tools

    if _nlp_tools:
        return _nlp_tools

    try:

        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer

        stop_words = set(stopwords.words("english"))
        stop_words.update(["waa", "iyo", "in", "uu", "ay", "ka", "u", "oo", "ah"])

        lemmatizer = WordNetLemmatizer()

        _nlp_tools = (word_tokenize, stop_words, lemmatizer)

    except Exception as e:

        print("NLP error:", e)
        _nlp_tools = (None, None, None)

    return _nlp_tools


# ================= TEXT PREPROCESS =================
def preprocess_text(text):

    word_tokenize, stop_words, lemmatizer = get_nlp_tools()

    if not word_tokenize:
        return text

    text = re.sub(r"[^a-z' ]", " ", text.lower())

    tokens = word_tokenize(text)

    clean = []

    for w in tokens:
        if w not in stop_words and len(w) > 2:
            clean.append(lemmatizer.lemmatize(w))

    return " ".join(clean)


# ================= URL CHECK =================
URL_PATTERN = re.compile(r"^(https?://|www\.)", re.I)

def is_url(text):
    return bool(URL_PATTERN.match(str(text).strip()))


# ================= SUBJECT =================
def guess_subject(text):

    t = text.lower()

    if "siyaasad" in t or "politics" in t:
        return "Politics"

    if "qarax" in t or "security" in t:
        return "Security"

    if "health" in t or "caafimaad" in t:
        return "Health"

    return "General"


# ================= URL SCRAPER =================
def extract_text_from_url(url):

    try:

        import requests
        from bs4 import BeautifulSoup

        r = requests.get(url, headers={"User-Agent": "Mozilla"}, timeout=10)

        soup = BeautifulSoup(r.content, "html.parser")

        for s in soup(["script", "style", "nav", "footer"]):
            s.decompose()

        body = " ".join(
            [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text()) > 30]
        )

        title = soup.title.string if soup.title else "Article"

        return body, title

    except Exception as e:

        print("URL extraction error:", e)

        return "", "Extraction Failed"


# ================= HEURISTIC =================
TRUSTED_SOURCES = [
    "bbc.com",
    "sonna.so",
    "sntv.so",
    "voasomali.com",
    "goobjoog.com",
    "hiiraan.com",
    "aljazeera.com",
]


def heuristic_fact_check(text, url=None):

    score = 0

    if url:

        if any(t in url.lower() for t in TRUSTED_SOURCES):
            score += 50

    if "!!!" in text:
        score -= 20

    if "deg deg" in text.lower():
        score -= 10

    if score >= 20:
        rating = "Trusted"

    elif score <= -10:
        rating = "Fake Information"

    else:
        rating = "Unverified"

    confidence = min(98, 50 + abs(score))

    return rating, confidence


# ================= DATASET SAVE =================
def add_to_dataset(text, label):

    try:

        import pandas as pd

        dataset_dir = os.path.join(BASE_DIR, "Dataset")

        os.makedirs(dataset_dir, exist_ok=True)

        file_name = "Real-news.csv" if label == "Trusted" else "Fake-news.csv"

        path = os.path.join(dataset_dir, file_name)

        row = pd.DataFrame(
            [
                {
                    "Text": text,
                    "label": 1 if label == "Trusted" else 0,
                }
            ]
        )

        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    except Exception as e:

        print("Dataset error:", e)


# ================= ROUTES =================

@app.route("/")
def home():
    return "Tafaftire Detection System Running"


@app.route("/api/health")
def health():
    return jsonify({"status": "OK"})


# ================= PREDICT =================
@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        global_stats["requests_handled"] += 1
        save_stats(global_stats)

        data = request.get_json(silent=True) or {}

        content = data.get("text") or data.get("data")

        if not content:
            return jsonify({"error": "No text"}), 400

        url = None
        title = "Article"

        if is_url(content):

            url = content if content.startswith("http") else "https://" + content

            content, title = extract_text_from_url(url)

            if not content:
                return jsonify({"error": "Failed to extract"}), 400

        import numpy as np

        model, vectorizer, _ = get_ml_resources()

        final_score = 0

        if model:

            X = vectorizer.transform([preprocess_text(content)]).toarray()

            final_score = model.decision_function(X)[0]

        rating, confidence = heuristic_fact_check(content, url)

        if rating == "Trusted":
            final_score += 2

        conf = min(98, max(70, (1 / (1 + np.exp(-abs(final_score)))) * 100))

        prediction = "Trusted" if final_score > -0.5 else "Fake Information"

        add_to_dataset(content, prediction)

        return jsonify(
            {
                "prediction": prediction,
                "confidence": f"{round(float(conf),2)}%",
                "title": title,
                "link": url or "N/A",
                "subject": guess_subject(content),
            }
        )

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ================= ADMIN =================
@app.route("/api/admin/login", methods=["POST"])
def admin_login():

    data = request.get_json() or {}

    if data.get("username") == "admin" and data.get("password") == "password123":

        return jsonify({"success": True, "token": "admin-token"})

    return jsonify({"success": False}), 401


@app.route("/api/admin/stats")
def admin_stats():

    def count_csv(name):

        p = os.path.join(BASE_DIR, "Dataset", name)

        try:
            with open(p, "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1
        except:
            return 0

    return jsonify(
        {
            "fake_news_count": count_csv("Fake-news.csv"),
            "real_news_count": count_csv("Real-news.csv"),
            "requests_handled": global_stats.get("requests_handled", 0),
            "system_status": "Healthy",
        }
    )


# ================= 404 =================
@app.errorhandler(404)
def not_found(e):

    if request.path.startswith("/api"):
        return jsonify({"error": "Not Found"}), 404

    return "Page Not Found"


# ================= RUN =================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 3402))

    print(f"Starting Tafaftire System on port {port}")

    app.run(host="0.0.0.0", port=port, debug=False)
