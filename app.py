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
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = os.environ.get("SECRET_KEY", "tafaftire-fallback-secret-key")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

STATS_FILE = os.path.join(DATA_DIR, "stats.json")
ANALYSIS_HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.json")
CONTACTS_FILE = os.path.join(DATA_DIR, "contacts.txt")

print(f"[*] DATA STORAGE: {DATA_DIR}")

# Startup Cleanup
for f in ["stats.json", "analysis_history.txt", "contacts.txt"]:
    if os.path.exists(f): 
        try: os.remove(f)
        except: pass

def save_analysis_result(original_input, confidence, label, extracted_text=None, data_type="AI Analysis", ai_score=None, expert_score=None, title="N/A", link="N/A", subject="General"):
    try:
        clean_input = str(original_input).strip()
        if not clean_input: return False
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        history = []
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except: history = []
        
        if history and history[-1].get("original_input") == clean_input: return True
                
        item_id = int(time.time() * 1000) + random.randint(1, 999)
        new_entry = {
            "id": item_id, "type": data_type, "original_input": clean_input,
            "extracted_text": extracted_text if extracted_text else clean_input,
            "confidence": confidence, "label": str(label), "date": timestamp,
            "ai_score": ai_score, "expert_score": expert_score,
            "title": title, "link": link or "N/A", "subject": subject
        }
        
        history.append(new_entry)
        if len(history) > 2000: history = history[-2000:]
            
        with open(ANALYSIS_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        add_to_dataset(text=extracted_text or clean_input, label=label, link=link, title=title, subject=subject)
        return True
    except Exception as e:
        print(f"[ERROR] History Save Failed: {e}")
        return False

def add_to_dataset(text, label, link="N/A", title="N/A", subject="General"):
    try:
        import pandas as pd
        if not text or len(str(text).strip()) < 10: return 
        
        label_str = str(label).upper()
        numerical_label = 1 if any(k in label_str for k in ["REAL", "TRUSTED", "RASMI", "RUN"]) else 0
        dataset_name = "Real-news.csv" if numerical_label == 1 else "fake-news.csv"
        
        path = os.path.join(os.path.dirname(__file__), "Dataset", dataset_name)
        if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
            
        # Duplicate check logic
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=['Text'], encoding='utf-8-sig')
                if str(text).strip().lower() in df['Text'].dropna().astype(str).str.strip().str.lower().tolist(): return
            except: pass

        new_row = pd.DataFrame([{"link": str(link)[:500], "title": str(title)[:200], "Text": str(text), "Subject": str(subject)[:100], "label": numerical_label}])
        new_row.to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"[ERROR] Dataset Update Failed: {e}")

def load_stats():
    defaults = {"requests_handled": 0, "model_accuracy": "94.5%"}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f: return {**defaults, **json.load(f)}
        except: pass
    return defaults

global_stats = load_stats()

# ================= NLTK SETUP =================
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try: nltk.data.find(pkg)
    except: nltk.download(pkg)

stop_words = set(stopwords.words("english"))
somali_stops = ["waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu", "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta", "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa", "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona"]
stop_words.update(somali_stops)
lemmatizer = WordNetLemmatizer()

# HELPERS
def preprocess_text(text):
    text = re.sub(r"[^a-z' ]", " ", str(text).lower())
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2])

def is_url(text):
    return bool(re.match(r'^(https?://|www\.)[a-z0-9-]+', str(text).strip(), re.I))

def extract_text_from_url(url):
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if resp.status_code != 200: return "", f"Error {resp.status_code}"
        soup = BeautifulSoup(resp.content, "html.parser")
        title = soup.title.string if soup.title else "News from Web"
        for s in soup(["script", "style", "nav"]): s.decompose()
        body = " ".join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2']) if len(p.get_text().split()) > 3])
        return body.strip(), title.strip()
    except Exception as e: return "", f"Error: {e}"

# ================= LOAD MODELS =================
model, vectorizer, label_encoder = None, None, None
try:
    import joblib
    import numpy as np
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl"))
    print("[OK] Models loaded successfully")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        import numpy as np
        data = request.get_json(silent=True) or {}
        content = data.get("text") or data.get("data")
        if not content: return jsonify({"error": "No text"}), 400

        input_url, page_title = None, "Article"
        if is_url(content):
            input_url = content if content.startswith("http") else "https://" + content
            content, page_title = extract_text_from_url(input_url)
        
        if not model: return jsonify({"error": "ML Model Offline"}), 503

        X = vectorizer.transform([preprocess_text(content)]).toarray()
        X_final = np.hstack([X, np.array([[0, 0]])]) 
        
        score = model.decision_function(X_final)[0]
        # Heuristic simple check
        trust_boost = 3.0 if any(t in str(input_url).lower() for t in ["bbc.com", "voasomali.com"]) else 0
        final_score = score + trust_boost
        conf = min(98.5, max(70.0, (1 / (1 + np.exp(-abs(final_score)))) * 100))
        
        return jsonify({
            "prediction": "Trusted" if final_score > -0.5 else "Fake Information",
            "confidence": f"{round(float(conf), 2)}%",
            "title": page_title, "link": input_url or "N/A"
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json() or {}
    if data.get("username") == "admin" and data.get("password") == "password123":
        return jsonify({"success": True, "token": "adm-token"})
    return jsonify({"success": False}), 401

if __name__ == "__main__":
    print("[*] Starting Tafaftire Server on Port 3402...")
    app.run(host="0.0.0.0", port=3402)
