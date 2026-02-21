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
app = Flask(__name__, 
            static_folder="Front_End_Data", 
            static_url_path="")
CORS(app)

# ================= NLTK OFFLINE SETUP =================
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

# Only download locally if not present
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR)

stop_words = set(stopwords.words("english"))
# Add Somali stopwords
somali_stopwords = ["waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", "aad", "ayaa"]
stop_words.update(somali_stopwords)
lemmatizer = WordNetLemmatizer()

# ================= HELPERS =================
def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        pass
    return text.strip()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()  # fallback if punkt missing
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def is_url(text):
    return bool(re.match(r'^(http|https)://', text.strip().lower()))

def extract_text_from_url(url):
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator=" ").strip()
    except Exception:
        return ""

# ================= LOAD MODELS =================
model, vectorizer, label_encoder = None, None, None
models_loaded = False
BASE_DIR = os.getcwd()
MODEL_FOLDER = os.path.join(BASE_DIR, "saved_model")

def load_models():
    global model, vectorizer, label_encoder, models_loaded
    try:
        # Fallback for case sensitivity
        target_folder = MODEL_FOLDER
        if not os.path.exists(target_folder):
            target_folder = os.path.join(BASE_DIR, "Saved_model")

        MODEL_PATH = os.path.join(target_folder, "svm_high_confidence.pkl")
        VECTORIZER_PATH = os.path.join(target_folder, "fake_real_TF_IDF_vectorizer.pkl")
        ENCODER_PATH = os.path.join(target_folder, "fake_real_label_encoder.pkl")

        if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]):
            print(f"❌ Model files not found in {target_folder}. Predictions will be disabled.")
            return

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        models_loaded = True
        print("✅ Models loaded successfully")

    except Exception as e:
        print("❌ Failed to load models:", e)
        traceback.print_exc()

load_models()

# ================= HEURISTIC FACT CHECKER (Preserved) =================
TRUSTED_SOURCES = [
    "bbc.com", "voasomali.com", "goobjoog.com", "garoweonline.com", 
    "somalistream.com", "somnn.com", "somaliglobe.net", "sntv.so", 
    "sonna.so", "aljazeera.com", "reuters.com", "apnews.com", 
    "hiiraan.com", "puntlandpost.net", "radiomuqdisho.net", "dalsan.so",
    "caasimada.net", "horseedmedia.net", "somalilandpost.net"
]

UNTRUSTED_PATTERNS = [
    "shidan", "fayras", "dawo mucjiso ah", "lacag bilaash ah", 
    "guji halkan", "win iphone", "naxdin", "deg deg", "nin yaaban",
    "naag yaaban", "subxaanallaah", "yaabka aduunka", "arrin lala yaabo",
    "qarax cusub", "war hadda soo dhacay", "daawasho naxdin leh",
    "lama rumeysan karo", "aad u naxdin badan", "muuqaal sir ah"
]

def heuristic_fact_check(text, url=None):
    score = 0
    reasons = []
    text_lower = text.lower()
    
    if url:
        url_lower = str(url).lower()
        clean_url = re.sub(r'^https?://(www\.)?', '', url_lower)
        for trusted in TRUSTED_SOURCES:
            if trusted in clean_url:
                score += 70 
                reasons.append(f"Isha rasmiga ah: {trusted}")
                break
    
    # Simple Somali Tone & Keyword Match
    professional_terms = ["madaxweyne", "baarlamaanka", "doorasho", "xukuumad", "amniga", "dowladda"]
    found_terms = [w for w in professional_terms if w in text_lower]
    score += len(found_terms) * 10

    if any(p in text_lower for p in UNTRUSTED_PATTERNS):
        score -= 30

    confidence = 55 + (abs(score) / 2)
    confidence = min(98, confidence)

    if score >= 15:
        rating = "Trusted"
    else:
        rating = "Unverified"
        confidence = max(50, confidence - 10)

    return {
        "rating": rating,
        "confidence": f"{int(confidence)}%",
        "reasons": reasons
    }

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return app.send_static_file("Index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK" if models_loaded else "ERROR",
        "models_loaded": models_loaded
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model-ada lama helin. Hubi folder-ka 'saved_model'."}), 503

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON data not found"}), 400

        input_type = data.get("type", "text")
        content = data.get("data", "")

        if not content:
            return jsonify({"error": "No text or URL provided"}), 400

        content = str(content).strip()

        if input_type == "url":
            if not is_url(content):
                return jsonify({"error": "Invalid URL"}), 400
            content = extract_text_from_url(content)
            if not content:
                return jsonify({"error": "Cannot extract text from URL"}), 400

        clean_input = preprocess_text(content)
        if not clean_input:
            return jsonify({"error": "Processed text is empty"}), 400

        X = vectorizer.transform([clean_input])
        expected_features = model.coef_.shape[1]

        # Fix feature dimension mismatch
        if X.shape[1] != expected_features:
            diff = expected_features - X.shape[1]
            if diff > 0:
                X = np.hstack([X.toarray(), np.zeros((X.shape[0], diff))])
            else:
                X = X.toarray()[:, :expected_features]
        else:
            X = X.toarray()

        pred = model.predict(X)[0]

        # Calculate confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            confidence = round(float(max(probs)) * 100, 2)
        else:
            score = model.decision_function(X)
            score = abs(score[0])
            confidence = round((1 / (1 + np.exp(-score))) * 100, 2)

        # Label matching for frontend ("Trusted" / "Fake Information")
        label = label_encoder.inverse_transform([pred])[0]
        result = "Trusted" if label == 1 or label == "REAL" or label == "Real" else "Fake Information"

        return jsonify({"prediction": result, "confidence": f"{confidence}%"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error occurred: {str(e)}"}), 500

@app.route("/fact-check", methods=["POST"])
def fact_check():
    try:
        data = request.get_json(silent=True)
        if not data: return jsonify({"error": "JSON error"}), 400
        content = data.get("data", "")
        input_url = content if is_url(content) else None
        
        if input_url:
            content = extract_text_from_url(input_url)
        
        if not content: return jsonify({"error": "Xog ma jirto"}), 400
        
        return jsonify(heuristic_fact_check(content, input_url))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data = request.get_json(silent=True)
        if not data: return jsonify({"error": "Data not found"}), 400
        name, email, message = data.get("name"), data.get("email"), data.get("message")
        if not all([name, email, message]): return jsonify({"error": "Please fill all fields"}), 400
        with open("contacts.txt", "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")
        return jsonify({"status": "Success", "message": "Fariintaada waa nala soo gaarsiiyey!"})
    except Exception:
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3402))
    app.run(host="0.0.0.0", port=port, debug=False)
