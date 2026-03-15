import os
import re
import traceback
import subprocess
import json
import time
import random
import csv
import smtplib
import imaplib
import email
import requests
import nltk
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Ensure terminal printing works on Windows for all characters
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ================= FLASK INIT =================
app = Flask(__name__, static_folder='Front_End', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.secret_key = os.environ.get("SECRET_KEY", "tafaftire-default-key-123")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Data Storage Setup
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

STATS_FILE = os.path.join(DATA_DIR, "stats.json")
ANALYSIS_HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.json")
CONTACTS_FILE = os.path.join(DATA_DIR, "contacts.txt")

# Startup Cleanup
for f in ["stats.json", "analysis_history.txt", "contacts.txt"]:
    if os.path.exists(f): 
        try: os.remove(f)
        except: pass

# ================= LAZY LOADING GLOBALS =================
_model = None
_vectorizer = None
_label_encoder = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_ml_resources():
    global _model, _vectorizer, _label_encoder
    if _model is None:
        try:
            import joblib
            model_path = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
            vec_path = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")
            enc_path = os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl")
            
            if os.path.exists(model_path):
                _model = joblib.load(model_path)
                _vectorizer = joblib.load(vec_path)
                _label_encoder = joblib.load(enc_path)
                print("[OK] Models loaded successfully")
            else:
                print("[!] Warning: Model files not found. App will use Heuristic check only.")
        except Exception as e:
            print(f"[ERROR] ML loading failed: {e}")
    return _model, _vectorizer, _label_encoder

# ================= HELPERS =================
def save_analysis_result(original_input, confidence, label, extracted_text=None, data_type="AI Analysis", ai_score=None, expert_score=None, title="N/A", link="N/A", subject="General"):
    try:
        clean_input = str(original_input).strip()
        if not clean_input: return False
        
        history = []
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except: history = []
        
        if history and history[-1].get("original_input") == clean_input: return True
                
        new_entry = {
            "id": int(time.time() * 1000) + random.randint(1, 999),
            "type": data_type, "original_input": clean_input,
            "extracted_text": extracted_text or clean_input,
            "confidence": confidence, "label": str(label), 
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ai_score": ai_score, "expert_score": expert_score,
            "title": title, "link": link or "N/A", "subject": subject
        }
        
        history.append(new_entry)
        with open(ANALYSIS_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[-2000:], f, indent=4)
            
        add_to_dataset(text=extracted_text or clean_input, label=label, link=link, title=title, subject=subject)
        return True
    except Exception as e:
        print(f"History error: {e}")
        return False

def add_to_dataset(text, label, **kwargs):
    try:
        import pandas as pd
        if not text or len(str(text).strip()) < 10: return 
        
        label_str = str(label).upper()
        is_real = any(k in label_str for k in ["REAL", "TRUSTED", "RASMI", "RUN"])
        dataset_name = "Real-news.csv" if is_real else "fake-news.csv"
        path = os.path.join(BASE_DIR, "Dataset", dataset_name)
        
        if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
            
        new_row = pd.DataFrame([{
            "link": str(kwargs.get('link', "N/A"))[:500],
            "title": str(kwargs.get('title', "N/A"))[:200],
            "Text": str(text),
            "Subject": str(kwargs.get('subject', "General"))[:100],
            "label": 1 if is_real else 0
        }])
        new_row.to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding='utf-8-sig')
    except Exception as e: print(f"Dataset loop error: {e}")

def load_stats():
    defaults = {"requests_handled": 0, "model_accuracy": "94.5%"}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f: return {**defaults, **json.load(f)}
        except: pass
    return defaults

def save_stats(stats):
    try:
        with open(STATS_FILE, "w") as f: json.dump(stats, f)
    except: pass

global_stats = load_stats()

# ================= NLTK SETUP =================
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try: nltk.data.find(pkg)
    except: nltk.download(pkg)

stop_words = set(stopwords.words("english"))
stop_words.update(["waa", "iyo", "in", "uu", "ay", "ka", "u", "ee", "oo", "ah"]) # Somalis simplified
lemmatizer = WordNetLemmatizer()

# PRE-COMPILED REGEX
URL_PATTERN = re.compile(r'^(https?://|www\.)', re.I)

def preprocess_text(text):
    text = re.sub(r"[^a-z' ]", " ", str(text).lower())
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2])

def is_url(text):
    return bool(URL_PATTERN.match(str(text).strip()))

def guess_subject(text):
    text_lower = str(text).lower()
    if any(w in text_lower for w in ["siyaasad", "doorasho", "politics"]): return "Politics"
    if any(w in text_lower for w in ["qarax", "amaanka", "security"]): return "Security"
    if any(w in text_lower for w in ["caafimaadka", "health"]): return "Health"
    return "General"

def extract_text_from_url(url):
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if resp.status_code != 200: return "", f"Error {resp.status_code}"
        soup = BeautifulSoup(resp.content, "html.parser")
        title = soup.title.string if soup.title else "News Article"
        for s in soup(["script", "style", "nav", "footer"]): s.decompose()
        body = " ".join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2']) if len(p.get_text().split()) > 3])
        return body.strip(), title.strip()
    except Exception as e: return "", f"Extraction Error: {e}"

# HEURISTIC
TRUSTED_SOURCES = ["bbc.com", "voasomali.com", "goobjoog.com", "hiiraan.com", "aljazeera.com"]

def heuristic_fact_check(text, url=None):
    score = 0
    reasons = []
    
    if url:
        if any(t in url.lower() for t in TRUSTED_SOURCES):
            score += 60
            reasons.append("Source is Trusted.")
    
    text_lower = text.lower()
    if any(p in text_lower for p in ["naxdin", "deg deg", "mucjiso"]):
        score -= 20
        reasons.append("Sensational language detected.")
        
    if "!!!" in text:
        score -= 15
        reasons.append("Excessive exclamation marks.")

    rating = "Trusted" if score >= 20 else ("Suspicious" if score <= -10 else "Unverified")
    confidence = min(98, 50 + (abs(score) / 2))
    
    return {"rating": rating, "confidence": f"{int(confidence)}%", "reasons": reasons, "score": score}

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return app.send_static_file('index.html')

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK"})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        global_stats["requests_handled"] += 1
        save_stats(global_stats)
        
        data = request.get_json(silent=True) or {}
        content = data.get("text") or data.get("data")
        if not content: return jsonify({"error": "No text"}), 400
        
        input_url, page_title = None, "Article"
        if is_url(content):
            input_url = content if content.startswith("http") else "https://" + content
            content, page_title = extract_text_from_url(input_url)
            if not content: return jsonify({"error": "Failed to extract from URL"}), 400

        model, vectorizer, _ = get_ml_resources()
        
        final_score = 0
        if model:
            X = vectorizer.transform([preprocess_text(content)]).toarray()
            X_final = np.hstack([X, np.array([[0, 0]])]) # Feature padding
            final_score = model.decision_function(X_final)[0]
        
        h_result = heuristic_fact_check(content, input_url)
        trust_boost = 3.0 if h_result["rating"] == "Trusted" else 0
        final_score += trust_boost
        
        conf = min(98.5, max(70.0, (1 / (1 + np.exp(-abs(final_score)))) * 100))
        
        return jsonify({
            "prediction": "Trusted" if final_score > -0.5 else "Fake Information",
            "confidence": f"{round(float(conf), 2)}%",
            "title": page_title, "link": input_url or "N/A",
            "subject": guess_subject(content)
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json() or {}
    if data.get("username") == "admin" and data.get("password") == "password123":
        return jsonify({"success": True, "token": "admin-session-token-xyz"})
    return jsonify({"success": False}), 401

@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    # Helper to count rows
    def count_csv(name):
        p = os.path.join(BASE_DIR, "Dataset", name)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f: return sum(1 for _ in f) - 1
        except: return 0
        
    return jsonify({
        "fake_news_count": count_csv("fake-news.csv"),
        "real_news_count": count_csv("Real-news.csv"),
        "requests_handled": global_stats.get("requests_handled", 0),
        "system_status": "Healthy"
    })

# Error handler for SPA
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api'): return jsonify({"error": "Not Found"}), 404
    return app.send_static_file('index.html')

# ================= RUN SERVER =================
if __name__ == "__main__":
    # Render requirements: Listen on $PORT
    port = int(os.environ.get("PORT", 3402))
    print(f"[*] Starting Tafaftire System on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
