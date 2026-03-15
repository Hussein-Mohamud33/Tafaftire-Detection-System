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

# Startup Cleanup
for f_name in ["stats.json", "analysis_history.txt", "contacts.txt"]:
    p = os.path.join(BASE_DIR, f_name)
    if os.path.exists(p):
        try: os.remove(p)
        except: pass

# ================= HISTORY & DATASET =================
def save_analysis_result(original_input, confidence, label, extracted_text=None, data_type="AI Analysis", ai_score=None, expert_score=None, title="N/A", link="N/A", subject="General"):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        i_id = int(time.time() * 1000)
        c_in = str(original_input).strip() or "N/A"
        if not extracted_text: extracted_text = c_in
            
        entry = {
            "id": i_id, "date": ts, "original_input": c_in, "extracted_text": extracted_text[:2000],
            "label": label, "confidence": confidence, "data_type": data_type,
            "ai_score": ai_score, "expert_score": expert_score, "title": title, "link": link, "subject": subject
        }

        history = []
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content: history = json.loads(content)
            except: pass
        
        history.insert(0, entry)
        history = history[:500]
        with open(ANALYSIS_HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(history, f, indent=4)
        add_to_dataset(extracted_text, label, link, title, subject)
        return True
    except: return False

def add_to_dataset(text, label, link="N/A", title="N/A", subject="General"):
    try:
        if not text or len(str(text).strip()) < 10: return 
        l_str = str(label).upper()
        d_name, num_l = ("Real-news.csv", 1) if any(k in l_str for k in ["REAL", "TRUSTED", "RASMI", "OFFICIAL"]) else ("Fake-news.csv", 0)
        p = os.path.join(BASE_DIR, "Dataset", d_name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        ex = os.path.exists(p)
        with open(p, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Title", "Text", "Category", "Label"])
            if not ex: w.writeheader()
            w.writerow({"Title": str(title)[:200], "Text": str(text), "Category": str(subject)[:100], "Label": num_l})
    except: pass

def load_stats():
    d = {"requests_handled": 0, "model_accuracy": "93.8%"}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f: return {**d, **json.load(f)}
        except: pass
    return d

def save_stats(s):
    try:
        with open(STATS_FILE, "w") as f: json.dump(s, f)
    except: pass

# ================= LAZY RESOURCES =================
nltk_initialized = False
model, vectorizer, label_encoder, lemmatizer, stop_words = None, None, None, None, set()
S_STOPS = ["waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu", "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta", "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa", "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona", "yahay", "yihiin", "ahayd", "ahaa", "noqday", "noqon", "leh", "leeyihiin", "hore", "danbe", "dhammaan", "kasta", "badnaa", "yar", "weyn", "waxa", "waxaa", "ila", "mid", "halkaas", "halkan", "door", "qaatay", "kaasoo", "ayadoo", "isagaa", "iyadaa", "kuwaasoo", "hadana", "maxaa", "maxay", "aynu", "idinku", "inay", "inuu", "una", "isuna", "isku"]

def load_resources(force=False):
    global model, vectorizer, label_encoder, lemmatizer, stop_words, nltk_initialized
    if nltk_initialized and not force: return
    import joblib, nltk
    d_dir = os.path.join(BASE_DIR, "nltk_data")
    os.makedirs(d_dir, exist_ok=True)
    if d_dir not in nltk.data.path: nltk.data.path.insert(0, d_dir)
    try:
        for r in ['tokenizers/punkt', 'tokenizers/punkt_tab', 'corpora/stopwords', 'corpora/wordnet']:
            try: nltk.data.find(r)
            except: nltk.download(r.split('/')[-1], download_dir=d_dir, quiet=True)
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        lemmatizer, stop_words = WordNetLemmatizer(), set(stopwords.words("english")).union(S_STOPS)
    except: stop_words = set(S_STOPS)
    try:
        model = joblib.load(os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl"))
        vectorizer = joblib.load(os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl"))
    except: pass
    nltk_initialized = True

def preprocess_text(t):
    load_resources()
    if not t: return ""
    from bs4 import BeautifulSoup
    from nltk.tokenize import word_tokenize
    t = BeautifulSoup(t, "html.parser").get_text().lower()
    t = re.sub(r"[^a-z' ]", " ", t)
    tokens = word_tokenize(t)
    return " ".join([lemmatizer.lemmatize(tk) if lemmatizer else tk for tk in tokens if tk not in stop_words and len(tk) > 2])

def extract_text_from_url(url):
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5, verify=False)
        s = BeautifulSoup(r.content, "html.parser")
        title = s.title.string if s.title else "News Article"
        for el in s(["script", "style", "header", "footer", "nav"]): el.decompose()
        txt = " ".join([p.get_text().strip() for p in s.find_all(['p', 'h1', 'h2']) if len(p.get_text().split()) > 5])
        return (txt if len(txt) > 100 else s.get_text(separator=" ", strip=True))[:8000], title.strip()
    except Exception as e: raise Exception(f"URL Error: {str(e)}")

def is_extreme_claim(t):
    words = ["100 sano", "mucjiso", "lacag bilaash", "mirecle", "cure", "hal charge"]
    return int(any(w in t.lower() for w in words))

def is_vague_source(t):
    words = ["khubaro ayaa sheegay", "warar la helayo", "ilo wareedyo", "daraasad ayaa sheegtay"]
    return int(any(w in t.lower() for w in words))

# ================= SEARCH & FACT CHECK =================
@lru_cache(maxsize=128)
def search_duckduckgo(q):
    try:
        from bs4 import BeautifulSoup
        res = requests.post("https://lite.duckduckgo.com/lite/", data={"q": q}, timeout=5.0)
        s = BeautifulSoup(res.text, 'html.parser')
        results = []
        for td in s.find_all('td', class_='result-snippet'):
            tr = td.find_parent('tr')
            if tr and tr.find_previous_sibling('tr'):
                a = tr.find_previous_sibling('tr').find('a', class_='result-link')
                if a: results.append({'snippet': td.text.strip(), 'link': a.get('href', ''), 'title': a.text.strip()})
        return results
    except: return []

TRUSTED = {"bbc.com", "voasomali.com", "garoweonline.com", "sntv.so", "sonna.so", "hiiraan.com", "aljazeera.com", "reuters.com"}
BAD_P = ["shidan", "mucjiso", "lacag bilaash", "guji halkan", "naxdin", "yaab", "fadeexad"]

def heuristic_fact_check(text, url=None):
    import tldextract
    s, r, t_l = 0, [], text.lower()
    is_t = False
    if url:
        try:
            ex = tldextract.extract(url); dom = f"{ex.domain}.{ex.suffix}".lower()
            if dom in TRUSTED: s += 100; is_t = True; r.append(f"Trusted: {dom}")
        except: pass
    load_resources()
    words = [w for w in text.split() if len(w) > 4 and w.lower() not in stop_words]
    q = " ".join(words[:6])
    if len(q) > 10:
        res = search_duckduckgo(q)
        m = sum(1 for rs in res[:5] if sum(1 for w in words[:8] if w.lower() in (rs['title']+rs['snippet']).lower()) >= 3)
        if m >= 2: s += 100; r.append("Multiple sources found.")
        elif m == 1: s += 40
    if sum(1 for p in BAD_P if p in t_l) >= 3: s -= 50; r.append("Sensationalist language.")
    conf = 60 + min(39, abs(s) * 0.4)
    if is_t: rating, lbl, conf = "Trusted", "OFFICIAL NEWS", max(95, conf)
    elif s >= 50: rating, lbl = "Trusted", "OFFICIAL NEWS"
    elif s <= -30: rating, lbl = "Fake", "FAKE NEWS"
    else: rating, lbl = "Unverified", "UNVERIFIED"
    return {"rating": rating, "label_so": lbl, "confidence": f"{int(conf)}%", "reasons": r}

# ================= ROUTES =================
@app.route("/", methods=["GET"])
@app.route("/admin", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def serve():
    if 'admin' in request.path.lower(): return app.send_static_file('Admin.html')
    return app.send_static_file('index.html')

@app.route("/api/health")
def health(): return jsonify({"status": "OK", "t": time.time()})

@app.route("/api/analyze_deep", methods=["POST"])
def analyze():
    try:
        import numpy as np
        load_resources(); d = request.get_json(silent=True) or {}
        orig = d.get("text") or d.get("data", "")
        if not orig: return jsonify({"error": "No content"}), 400
        u = orig if bool(re.match(r'^(https?://|www\.)', orig)) else None
        if u and not u.startswith("http"): u = "https://" + u
        content, title = extract_text_from_url(u) if u else (orig, orig[:60]+"...")
        X = vectorizer.transform([preprocess_text(content)])
        
        # [FIX] Added Meta-features to match training (12000 + 2 = 12002)
        X = np.hstack([X.toarray(), np.array([[is_extreme_claim(content), is_vague_source(content)]])])
        
        raw = model.decision_function(X)[0] if model else 0
        ai_p, ai_c = ("Real News" if raw >= 0 else "Fake news"), (1 / (1 + np.exp(-abs(raw * 2.0)))) * 100
        if len(content.split()) < 20 and -0.8 < raw < 0: ai_p = "Real News"
        fc = heuristic_fact_check(content, u); fcn = float(fc["confidence"].replace("%", ""))
        final, winning_c, src = (("OFFICIAL NEWS" if ai_p == "Real News" else "FAKE NEWS"), f"{ai_c:.1f}%", "AI Engine") if ai_c >= fcn else (fc["label_so"], fc["confidence"], "Expert Logic")
        if len(content.split()) < 30 and final == "FAKE NEWS" and max(ai_c, fcn) < 95: final = "SUSPICIOUS"
        save_analysis_result(orig, winning_c, final, content, src, f"{ai_c:.1f}%", fc["confidence"], title, u or "N/A", guess_subject(content))
        return jsonify({"final_verdict": final, "label_so": final, "winning_confidence": winning_c, "winning_source": src, "ai_res": {"prediction": ai_p, "confidence": f"{ai_c:.1f}%"}, "fc_res": fc, "title": title, "status": "success"})
    except Exception as e: return jsonify({"error": str(e)}), 500

# ================= ADMIN =================
ADMIN_T = "admin-session-token-123"
def admin_req(f):
    @wraps(f)
    def dec(*args, **kwargs):
        if request.headers.get("Authorization") != ADMIN_T: return jsonify({"success": False}), 401
        return f(*args, **kwargs)
    return dec

@app.route("/api/admin/login", methods=["POST"])
def login():
    d = request.get_json()
    if d.get("username") == "admin" and d.get("password") == "password123": return jsonify({"success": True, "token": ADMIN_T})
    return jsonify({"success": False}), 401

@app.route("/api/admin/stats")
@admin_req
def stats():
    st = load_stats(); h_c = 0
    if os.path.exists(ANALYSIS_HISTORY_FILE):
        try:
            with open(ANALYSIS_HISTORY_FILE, "r") as f: h_c = len(json.load(f))
        except: pass
    msg_c = 0
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, "r") as f: msg_c = f.read().count("---\n")
    return jsonify({"requests_handled": st.get("requests_handled", 0), "history_count": h_c, "model_accuracy": st["model_accuracy"], "total_datasets": 2, "system_status": "Healthy", "messages_count": msg_c})

@app.route("/api/admin/analysis_history")
@admin_req
def history():
    if not os.path.exists(ANALYSIS_HISTORY_FILE): return jsonify([])
    with open(ANALYSIS_HISTORY_FILE, "r") as f: return jsonify(json.load(f))

@app.route("/api/admin/analysis_history/delete", methods=["POST"])
@admin_req
def del_hist():
    i_id = request.get_json().get("id")
    if not os.path.exists(ANALYSIS_HISTORY_FILE): return jsonify({"success": False}), 404
    with open(ANALYSIS_HISTORY_FILE, "r") as f: h = json.load(f)
    with open(ANALYSIS_HISTORY_FILE, "w") as f: json.dump([i for i in h if i.get("id") != i_id], f, indent=4)
    return jsonify({"success": True})

@app.route("/api/admin/analysis_history/clear", methods=["POST"])
@admin_req
def clear_hist():
    with open(ANALYSIS_HISTORY_FILE, "w") as f: json.dump([], f)
    return jsonify({"success": True})

@app.route("/api/admin/datasets")
@admin_req
def datasets():
    dp = os.path.join(BASE_DIR, "Dataset")
    if not os.path.exists(dp): os.makedirs(dp)
    return jsonify([{"name": f, "size": f"{os.path.getsize(os.path.join(dp, f))/1024:.1f} KB", "modified": time.ctime(os.path.getmtime(os.path.join(dp, f)))} for f in os.listdir(dp) if f.endswith(".csv")])

@app.route("/api/admin/dataset/get")
@admin_req
def get_ds():
    import csv
    p = os.path.join(BASE_DIR, "Dataset", request.args.get("filename"))
    with open(p, "r", encoding="utf-8-sig") as f: r = list(csv.reader(f))
    return jsonify({"columns": r[0] if r else [], "data": r[1:] if len(r)>1 else []})

@app.route("/api/admin/dataset/save", methods=["POST"])
@admin_req
def save_ds():
    import csv
    d = request.get_json()
    p = os.path.join(BASE_DIR, "Dataset", d["filename"])
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(d["columns"]); w.writerows(d["rows"])
    return jsonify({"success": True})

@app.route("/api/admin/retrain", methods=["POST"])
@admin_req
def retrain():
    import subprocess, sys
    subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "Model_trains.py")])
    return jsonify({"success": True})

@app.route("/api/admin/logs")
@admin_req
def logs():
    m = []
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, "r") as f:
            for p in f.read().split("---\n"):
                if p.strip():
                    it = {}
                    for l in p.strip().split("\n"):
                        if ":" in l: k,v = l.split(":",1); it[k.lower().strip()] = v.strip()
                    if it: m.append(it)
    return jsonify(m[::-1])

@app.route("/api/admin/logs/delete", methods=["POST"])
@admin_req
def del_log():
    l_id = request.get_json().get("id")
    if not os.path.exists(CONTACTS_FILE): return jsonify({"success": False}), 404
    with open(CONTACTS_FILE, "r") as f: pts = f.read().split("---\n")
    with open(CONTACTS_FILE, "w") as f: f.write("".join([p.strip() + "\n---\n" for i, p in enumerate(pts) if i != l_id and p.strip()]))
    return jsonify({"success": True})

@app.route("/api/admin/sync_emails", methods=["POST"])
@admin_req
def sync_emails():
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login("tafaftiredetectionsystem@gmail.com", "qgzpeswwwgtgawuy")
        mail.select('inbox'); _, ms = mail.search(None, 'UNSEEN'); cnt = 0
        for e_id in ms[0].split():
            _, md = mail.fetch(e_id, '(RFC822)'); msg = email.message_from_bytes(md[0][1])
            body = msg.get_payload(decode=True).decode() if not msg.is_multipart() else ""
            with open(CONTACTS_FILE, "a", encoding="utf-8") as f: f.write(f"Name: {msg.get('From')}\nEmail: {msg.get('From')}\nMessage: [EMAIL] {body}\n---\n")
            mail.store(e_id, '+FLAGS', '\\Seen'); cnt += 1
        mail.logout()
        return jsonify({"success": True, "count": cnt})
    except: return jsonify({"success": False}), 500

@app.route("/api/admin/reply", methods=["POST"])
@admin_req
def reply():
    d = request.get_json()
    r, s, b = d.get("email"), d.get("subject"), d.get("body")
    try:
        msg = MIMEMultipart(); msg['From'], msg['To'], msg['Subject'] = "Tafaftire <tafaftiredetectionsystem@gmail.com>", r, s
        msg.attach(MIMEText(b, 'plain'))
        srv = smtplib.SMTP('smtp.gmail.com', 587); srv.starttls(); srv.login("tafaftiredetectionsystem@gmail.com", "qgzpeswwwgtgawuy")
        srv.send_message(msg); srv.quit()
        return jsonify({"success": True})
    except: return jsonify({"success": False}), 500

@app.route("/contact", methods=["POST"])
def contact():
    d = request.get_json()
    with open(CONTACTS_FILE, "a") as f: f.write(f"Name: {d.get('name')}\nEmail: {d.get('email')}\nMessage: {d.get('message')}\n---\n")
    return jsonify({"status": "Success"})

if __name__ == "__main__":
    import sys
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3402)), debug=False)
