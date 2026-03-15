import os
import re
import traceback
import requests
import json
import time
import csv
import smtplib
import imaplib
import email
import datetime
import joblib
import numpy as np
import nltk
import tldextract
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import lru_cache, wraps

# ================= SYSTEM ARCHITECTURE (QAAB-DHISMEEDKA SYSTEM-KA) =================
# 1. FRONTEND: (Front_End/index.html, script.js) - UI-ga uu isticmaalaha arko.
# 2. BACKEND: (app.py) - Flask API oo xiriirisa AI iyo Fact-check-ka.
# 3. AI ENGINE: (saved_model/) - SVM Model oo lagu tababaray kumanaan warar Somali/English ah.
# 4. EXPERT LAYER: (heuristic_fact_check) - Baaritaan Live ah oo internet-ka ah (DuckDuckGo).
# 5. DATA STORAGE: (~/.tafaftire_system_data) - Meesha ay ku kaydsan yihiin tariikhda iyo xogta.
# ===================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder='Front_End', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 

CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "Authorization"],
    "methods": ["GET", "POST", "OPTIONS"]
}}, supports_credentials=True)

HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")
os.makedirs(DATA_DIR, exist_ok=True)

STATS_FILE = os.path.join(DATA_DIR, "stats.json")
ANALYSIS_HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.json")
CONTACTS_FILE = os.path.join(DATA_DIR, "contacts.txt")

# Startup Cleanup: Remove legacy files from root workspace if they exist
for legacy_file in ["stats.json", "analysis_history.txt", "contacts.txt"]:
    path = os.path.join(BASE_DIR, legacy_file)
    if os.path.exists(path):
        try: os.remove(path)
        except: pass

# ================= HISTORY & DATASET =================
def save_analysis_result(original_input, confidence, label, extracted_text=None, data_type="AI Analysis", ai_score=None, expert_score=None, title="N/A", link="N/A", subject="General"):
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        item_id = int(time.time() * 1000)
        clean_input = str(original_input).strip() if original_input else "N/A"
        if not extracted_text: extracted_text = clean_input
            
        new_entry = {
            "id": item_id, "date": timestamp, "original_input": clean_input,
            "extracted_text": extracted_text[:2000], "label": label,
            "confidence": confidence, "data_type": data_type,
            "ai_score": ai_score, "expert_score": expert_score,
            "title": title, "link": link, "subject": subject
        }

        history = []
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content: history = json.loads(content)
            except: pass
        
        history.insert(0, new_entry)
        history = history[:500]
        
        with open(ANALYSIS_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        
        add_to_dataset(text=extracted_text, label=label, link=link, title=title, subject=subject)
        return True
    except:
        print(f"[!] Error in save_analysis_result: {traceback.format_exc()}")
        return False

def add_to_dataset(text, label, link="N/A", title="N/A", subject="General"):
    try:
        if not text or len(str(text).strip()) < 10: return 
        label_str = str(label).upper()
        dataset_name = "Fake-news.csv"
        numerical_label = 0
        
        if any(keyword in label_str for keyword in ["REAL", "TRUSTED", "RASMI", "OFFICIAL", "RUN"]):
            dataset_name = "Real-news.csv"
            numerical_label = 1

        dataset_path = os.path.join(BASE_DIR, "Dataset", dataset_name)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        file_exists = os.path.exists(dataset_path)
        with open(dataset_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Title", "Text", "Category", "Label"])
            if not file_exists: writer.writeheader()
            writer.writerow({
                "Title": str(title)[:200], "Text": str(text),
                "Category": str(subject)[:100], "Label": numerical_label
            })
    except Exception as e:
        print(f"[!] Dataset update failed: {e}")

def load_stats():
    defaults = {"requests_handled": 0, "model_accuracy": "94.5%"}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                return {**defaults, **json.load(f)}
        except: pass
    return defaults

def save_stats(stats):
    try:
        with open(STATS_FILE, "w") as f: json.dump(stats, f)
    except: pass

_global_stats_cache = None
def get_global_stats():
    global _global_stats_cache
    if _global_stats_cache is None: _global_stats_cache = load_stats()
    return _global_stats_cache

# ================= NLTK & MODEL STATE =================
nltk_initialized = False
model = None
vectorizer = None
label_encoder = None
lemmatizer = None
stop_words = set()
SOMALI_STOPWORDS = [
    "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
    "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
    "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
    "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
    "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona",
    "yahay", "yihiin", "ahayd", "ahaa", "noqday", "noqon", "leh", "leeyihiin",
    "kala", "hore", "danbe", "dhammaan", "kasta", "badnaa", "yar", "weyn",
    "waxa", "waxaa", "ila", "mid", "loo", "halkaas", "halkan", "door", "qaatay",
    "kaasoo", "ayadoo", "isagaa", "iyadaa", "kuwaasoo", "hadana", "maxaa", "maxay",
    "aynu", "idinku", "inay", "inuu", "loogu", "una", "isuna", "isku"
]

def load_resources(force=False):
    global model, vectorizer, label_encoder, lemmatizer, stop_words, nltk_initialized
    if nltk_initialized and not force: return

    data_dir = os.path.join(BASE_DIR, "nltk_data")
    os.makedirs(data_dir, exist_ok=True)
    if data_dir not in nltk.data.path: nltk.data.path.insert(0, data_dir)

    try:
        for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab', 'corpora/stopwords', 'corpora/wordnet']:
            try: nltk.data.find(resource)
            except LookupError: nltk.download(resource.split('/')[-1], download_dir=data_dir)

        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english")).union(SOMALI_STOPWORDS)
    except: stop_words = set(SOMALI_STOPWORDS)

    try:
        model = joblib.load(os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl"))
        vectorizer = joblib.load(os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl"))
    except Exception as e: print(f"[!] Model load failure: {e}")
    nltk_initialized = True

# ================= HELPERS =================
URL_PATTERN = re.compile(r'^(https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+([/?#].*)?$', re.IGNORECASE)
CLEAN_TEXT_PATTERN = re.compile(r"[^a-z' ]")

def preprocess_text(text):
    load_resources()
    if not text: return ""
    text = BeautifulSoup(text, "html.parser").get_text().lower()
    text = CLEAN_TEXT_PATTERN.sub(" ", text)
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(t) if lemmatizer else t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(cleaned)

def is_url(text): return bool(URL_PATTERN.match(text.strip()))

def guess_subject(text):
    text_lower = text.lower()
    mapping = {
        "Siyaasadda": ["siyaasad", "baarlaman", "doorasho", "government", "policy", "maamulka", "xilka"],
        "Amniga": ["qarax", "amaanka", "ciidanka", "police", "security", "dagaal", "killed", "shil"],
        "Caafimaadka": ["caafimaadka", "isbitaal", "health", "doctor", "virus", "fayras", "dawo"],
        "Dhaqaalaha": ["lacag", "dhaqaale", "bank", "finance", "economy", "ganacsi", "cashuur", "deynta"]
    }
    for sub, keys in mapping.items():
        if any(k in text_lower for k in keys): return sub
    return "Guud"

def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5, verify=False)
        if resp.status_code != 200: raise Exception(f"HTTP {resp.status_code}")
        
        soup = BeautifulSoup(resp.content, "html.parser")
        page_title = soup.title.string if soup.title else "News Article"
        for el in soup(["script", "style", "header", "footer", "nav"]): el.decompose()
        
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().split()) > 5])
        
        if len(text) < 100: text = soup.get_text(separator=" ", strip=True)
        return text[:10000], page_title.strip()
    except Exception as e: raise Exception(f"URL Extract Error: {str(e)}")

# ================= SEARCH & FACT CHECK =================
@lru_cache(maxsize=128)
def search_duckduckgo(query):
    try:
        res = requests.post("https://lite.duckduckgo.com/lite/", data={"q": query}, timeout=5.0)
        soup = BeautifulSoup(res.text, 'html.parser')
        results = []
        for td in soup.find_all('td', class_='result-snippet'):
            tr = td.find_parent('tr')
            if tr:
                prev = tr.find_previous_sibling('tr')
                if prev:
                    a = prev.find('a', class_='result-link')
                    if a: results.append({'snippet': td.text.strip(), 'link': a.get('href', ''), 'title': a.text.strip()})
        return results
    except: return []

TRUSTED_SOURCES = {
    "bbc.com", "voasomali.com", "goobjoog.net", "garoweonline.com", "sntv.so", "sonna.so", "hiiraan.com",
    "caasimada.net", "jowhar.com", "radiomuqdisho.net", "villasomalia.gov.so", "aljazeera.com", "reuters.com"
}
UNTRUSTED_PATTERNS = ["shidan", "mucjiso", "lacag bilaash", "guji halkan", "naxdin", "yaab", "fadeexad", "si degdeg ah"]

def heuristic_fact_check(text, url=None):
    score, reasons = 0, []
    text_lower = text.lower()
    untrusted_matches = sum(1 for p in UNTRUSTED_PATTERNS if p in text_lower)
    is_trusted_domain = False

    if url:
        try:
            ext = tldextract.extract(url)
            domain = f"{ext.domain}.{ext.suffix}".lower()
            if domain in TRUSTED_SOURCES:
                score += 100
                is_trusted_domain = True
                reasons.append(f"Trusted Source: {domain}")
        except: pass

    meaningful_words = [w for w in text.split() if len(w) > 4 and w.lower() not in stop_words]
    query = " ".join(meaningful_words[:6])
    
    found_citations = False
    if len(query) > 10:
        results = search_duckduckgo(query)
        match_count = 0
        for res in results[:5]:
            res_content = (res['title'] + " " + res['snippet']).lower()
            matches = sum(1 for w in meaningful_words[:8] if w.lower() in res_content)
            if matches >= 3: match_count += 1
        
        if match_count >= 2:
            score += 100
            found_citations = True
            reasons.append("Multiple sources confirmed this context.")
        elif match_count == 1: score += 40

    if untrusted_matches >= 3:
        score -= 50
        reasons.append(f"Sensationalist language detected ({untrusted_matches} matches)")

    conf = 60 + min(39, abs(score) * 0.4)
    if is_trusted_domain: rating, label, conf = "Trusted", "OFFICIAL NEWS", max(95, conf)
    elif score >= 50: rating, label = "Trusted", "OFFICIAL NEWS"
    elif score <= -30: rating, label = "Fake", "FAKE NEWS"
    else: rating, label = "Unverified", "UNVERIFIED"

    return {"rating": rating, "label_so": label, "confidence": f"{int(conf)}%", "reasons": reasons}

# ================= ROUTES =================
@app.route("/", methods=["GET"])
@app.route("/admin", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def serve_index():
    path = request.path.lower()
    if 'admin' in path: return app.send_static_file('Admin.html')
    return app.send_static_file('index.html')

@app.route("/api/health")
def health(): return jsonify({"status": "OK", "uptime": time.time()})

@app.route("/api/analyze_deep", methods=["POST"])
def analyze_deep():
    try:
        load_resources()
        gs = get_global_stats()
        gs["requests_handled"] += 1
        save_stats(gs)

        data = request.get_json(silent=True) or {}
        content = data.get("text") or data.get("data", "")
        if not content: return jsonify({"error": "No content"}), 400

        input_url = content if is_url(content) else None
        if input_url and not input_url.startswith("http"): input_url = "https://" + input_url
        
        title = "News Article"
        if input_url: content, title = extract_text_from_url(input_url)
        else: title = content[:60] + "..."

        # AI Prediction
        X = vectorizer.transform([preprocess_text(content)])
        raw_score = model.decision_function(X.toarray())[0]
        ai_pred = "Real News" if raw_score >= 0 else "Fake news"
        ai_conf_num = (1 / (1 + np.exp(-abs(raw_score * 2.0)))) * 100
        
        word_count = len(content.split())
        if word_count < 20 and -0.8 < raw_score < 0: ai_pred = "Real News" # Short text bias
        
        # Expert Check
        fc_res = heuristic_fact_check(content, input_url)
        fc_conf_num = float(fc_res["confidence"].replace("%", ""))

        # Consensus
        if ai_conf_num >= fc_conf_num:
            final_label = "OFFICIAL NEWS" if ai_pred == "Real News" else "FAKE NEWS"
            winning_conf = f"{ai_conf_num:.2f}%"
            source = "AI Analysis Engine"
        else:
            final_label = fc_res["label_so"]
            winning_conf = fc_res["confidence"]
            source = "Expert Verification Logic"

        if word_count < 30 and final_label == "FAKE NEWS" and max(ai_conf_num, fc_conf_num) < 95:
            final_label = "SUSPICIOUS"

        save_analysis_result(data.get("text") or data.get("data"), winning_conf, final_label, content, source, f"{ai_conf_num:.1f}%", fc_res["confidence"], title, input_url or "N/A", guess_subject(content))

        return jsonify({
            "final_verdict": final_label, "label_so": final_label, "winning_confidence": winning_conf,
            "winning_source": source, "ai_res": {"prediction": ai_pred, "confidence": f"{ai_conf_num:.1f}%"},
            "fc_res": fc_res, "title": title, "status": "success"
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

# ================= ADMIN ACTIONS =================
ADMIN_TOKEN = "admin-session-token-123"

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get("Authorization") != ADMIN_TOKEN:
            return jsonify({"success": False, "message": "Auth Required"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    if data.get("username") == "admin" and data.get("password") == "password123":
        return jsonify({"success": True, "token": ADMIN_TOKEN})
    return jsonify({"success": False}), 401

@app.route("/api/admin/stats")
@admin_required
def admin_stats():
    gs = get_global_stats()
    history_count = 0
    if os.path.exists(ANALYSIS_HISTORY_FILE):
        with open(ANALYSIS_HISTORY_FILE, "r") as f: history_count = len(json.load(f))
    
    return jsonify({
        "requests_handled": gs["requests_handled"],
        "history_count": history_count,
        "model_accuracy": gs["model_accuracy"],
        "total_datasets": 2,
        "system_status": "Healthy"
    })

@app.route("/api/admin/analysis_history")
@admin_required
def get_history():
    if not os.path.exists(ANALYSIS_HISTORY_FILE): return jsonify([])
    with open(ANALYSIS_HISTORY_FILE, "r") as f: return jsonify(json.load(f))

@app.route("/api/admin/analysis_history/delete", methods=["POST"])
@admin_required
def delete_history():
    item_id = request.get_json().get("id")
    with open(ANALYSIS_HISTORY_FILE, "r") as f: history = json.load(f)
    history = [i for i in history if i["id"] != item_id]
    with open(ANALYSIS_HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)
    return jsonify({"success": True})

@app.route("/api/admin/analysis_history/clear", methods=["POST"])
@admin_required
def clear_history():
    with open(ANALYSIS_HISTORY_FILE, "w") as f: json.dump([], f)
    return jsonify({"success": True})

@app.route("/api/admin/analysis_history/sync_all", methods=["POST"])
@admin_required
def sync_all():
    if not os.path.exists(ANALYSIS_HISTORY_FILE): return jsonify({"success": False}), 404
    with open(ANALYSIS_HISTORY_FILE, "r") as f: history = json.load(f)
    for item in history:
        text = item.get("extracted_text") or item.get("original_input")
        add_to_dataset(text, item.get("label"), link=item.get("link", "N/A"), title=item.get("title", "Historical"), subject=item.get("subject", "General"))
    return jsonify({"success": True, "count": len(history)})

@app.route("/api/admin/datasets")
@admin_required
def list_datasets():
    dir_path = os.path.join(BASE_DIR, "Dataset")
    files = []
    for f in os.listdir(dir_path):
        if f.endswith(".csv"):
            p = os.path.join(dir_path, f)
            files.append({"name": f, "size": f"{os.path.getsize(p)/1024:.1f} KB", "modified": time.ctime(os.path.getmtime(p))})
    return jsonify(files)

@app.route("/api/admin/dataset/get")
@admin_required
def get_dataset():
    path = os.path.join(BASE_DIR, "Dataset", request.args.get("filename"))
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return jsonify({"columns": rows[0], "data": rows[1:]})

@app.route("/api/admin/dataset/save", methods=["POST"])
@admin_required
def save_dataset():
    data = request.get_json()
    path = os.path.join(BASE_DIR, "Dataset", data["filename"])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data["columns"])
        writer.writerows(data["rows"])
    return jsonify({"success": True})

@app.route("/api/admin/retrain", methods=["POST"])
@admin_required
def retrain():
    script = os.path.join(BASE_DIR, "Model_trains.py")
    subprocess.Popen([sys.executable, script])
    return jsonify({"success": True})

@app.route("/api/admin/reload_models", methods=["POST"])
@admin_required
def reload_models():
    load_resources(force=True)
    return jsonify({"success": True})

@app.route("/api/admin/logs")
@admin_required
def list_logs():
    messages = []
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, "r") as f:
            for part in f.read().split("---\n"):
                if part.strip():
                    item = {}
                    for line in part.strip().split("\n"):
                        if ":" in line: k,v = line.split(":",1); item[k.lower().strip()] = v.strip()
                    messages.append(item)
    return jsonify(messages[::-1])

@app.route("/contact", methods=["POST"])
def contact():
    data = request.get_json()
    with open(CONTACTS_FILE, "a") as f:
        f.write(f"Name: {data['name']}\nEmail: {data['email']}\nMessage: {data['message']}\n---\n")
    return jsonify({"status": "Success"})

@app.route("/api/admin/reply", methods=["POST"])
@admin_required
def admin_reply():
    data = request.get_json()
    recipient, subject, body = data.get("email"), data.get("subject"), data.get("body")
    if not all([recipient, subject, body]): return jsonify({"success": False}), 400

    sender_email = "tafaftiredetectionsystem@gmail.com"
    sender_password = "qgzpeswwwgtgawuy"

    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = f"Tafaftire <{sender_email}>", recipient, subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/sync_emails", methods=["POST"])
@admin_required
def sync_emails():
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login("tafaftiredetectionsystem@gmail.com", "qgzpeswwwgtgawuy")
        mail.select('inbox')
        status, messages = mail.search(None, 'UNSEEN')
        
        count = 0
        for e_id in messages[0].split():
            res, msg_data = mail.fetch(e_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg['Subject'])[0][0]
                    if isinstance(subject, bytes): subject = subject.decode()
                    from_ = msg.get('From')
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode()
                                break
                    else: body = msg.get_payload(decode=True).decode()
                    
                    with open(CONTACTS_FILE, "a", encoding="utf-8") as f:
                        f.write(f"Name: {from_}\nEmail: {from_}\nMessage: [EMAIL] {subject} - {body}\n---\n")
                    mail.store(e_id, '+FLAGS', '\\Seen')
                    count += 1
        mail.logout()
        return jsonify({"success": True, "count": count})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/logs/delete", methods=["POST"])
@admin_required
def delete_log():
    log_id = request.get_json().get("id")
    with open(CONTACTS_FILE, "r") as f: parts = f.read().split("---\n")
    new_parts = [p.strip() + "\n---\n" for idx, p in enumerate(parts) if idx != log_id and p.strip()]
    with open(CONTACTS_FILE, "w") as f: f.write("".join(new_parts))
    return jsonify({"success": True})

if __name__ == "__main__":
    import sys
    import subprocess
    port = int(os.environ.get("PORT", 3402))
    app.run(host="0.0.0.0", port=port, debug=False)
