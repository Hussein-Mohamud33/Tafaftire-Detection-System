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

# ================= NLTK SETUP (Render Safe) =================
nltk_data_path = "/opt/render/nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

nltk.data.path.append(nltk_data_path)

stop_words = set(stopwords.words("english"))
somali_stopwords = [
    "waa","iyo","in","uu","ay","ayuu","ayey","ka","u","ee","oo","ah",
    "sidii","waxaan","waxaad","wuxuu","waxay","iska","ahaa","lagu","loogu",
    "isagoo","iyadoo","ku","soo","isaga","iyada","labada","kala","inta",
    "ilaa","wax","kale","mar","markii","la","si","aad","eeg","ayaa",
    "ayay","kuwa","kuwaas","kuwan","kaas","kan","kuwaa","loo","loona"
]
stop_words.update(somali_stopwords)

lemmatizer = WordNetLemmatizer()

# ================= HELPERS =================
def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.strip()

def preprocess_text(text):
    text = sanitize_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

def is_url(text):
    pattern = r'^(https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+([/?#].*)?$'
    return bool(re.match(pattern, text.strip().lower()))

def extract_text_from_url(url):
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator=" ").strip()
    except Exception:
        return ""

# ================= HEURISTIC CHECK =================
TRUSTED_SOURCES = [
    "bbc.com","voasomali.com","goobjoog.com","garoweonline.com",
    "somalistream.com","somnn.com","somaliglobe.net","sntv.so",
    "sonna.so","aljazeera.com","reuters.com","apnews.com","hiiraan.com"
]

def heuristic_fact_check(text, url=None):
    score = 0
    reasons = []
    text_lower = text.lower()

    if url:
        clean_url = re.sub(r'^https?://(www\.)?', '', url.lower())
        if any(t in clean_url for t in TRUSTED_SOURCES):
            score += 40
            reasons.append("Isha warka waa mid la yaqaan.")

    if len(text.split()) < 30:
        score -= 15
        reasons.append("Qoraalka waa gaaban.")

    confidence = 50 + abs(score)
    confidence = min(95, confidence)

    rating = "Trusted" if score >= 10 else "Unverified"

    return {
        "rating": rating,
        "confidence": f"{confidence}%",
        "reasons": reasons
    }

# ================= LOAD MODEL =================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("✅ Models loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
    traceback.print_exc()
    exit(1)

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "API is running"})

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
        input_url = None

        # URL Handling
        if is_url(content):
            if not content.startswith(("http://","https://")):
                content = "https://" + content
            input_url = content
            extracted = extract_text_from_url(input_url)
            if not extracted:
                return jsonify({"error": "URL laga ma soo saari karin xog"}), 400
            content = extracted

        # Preprocess
        clean_input = preprocess_text(content)

        # VECTORIZE ONLY (NO EXTRA FEATURES)
        X = vectorizer.transform([clean_input])

        # Model Score
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
        else:
            score = float(model.predict(X)[0])

        # Heuristic boost (NO feature stacking)
        trust_boost = 0.0
        if input_url:
            h = heuristic_fact_check(content, input_url)
            if h["rating"] == "Trusted":
                trust_boost += 1.5

        final_score = score + trust_boost

        confidence = (1 / (1 + np.exp(-abs(final_score)))) * 100
        confidence = min(98.0, max(70.0, confidence))

        result = "Trusted" if final_score > 0 else "Fake Information"

        return jsonify({
            "prediction": result,
            "confidence": f"{round(confidence,2)}%"
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error ayaa dhacay"}), 500

@app.route("/fact-check", methods=["POST"])
def fact_check():
    try:
        data = request.get_json(silent=True)
        content = data.get("text")
        if not content:
            return jsonify({"error": "Xog lama helin"}), 400

        result = heuristic_fact_check(content)
        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error"}), 500

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Server starting on port {port}")
    app.run(host="0.0.0.0", port=port)
