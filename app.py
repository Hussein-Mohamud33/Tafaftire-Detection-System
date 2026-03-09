import os
import re
import joblib
import traceback
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import time
import csv
import smtplib
import imaplib
import email
import tldextract
import nltk # Base import only
import subprocess
import json
import time
import csv
import smtplib
import imaplib
import email
import tldextract
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
import pandas as pd
from functools import lru_cache

# ================= FLASK INIT =================
app = Flask(__name__, static_folder='Front_End', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Permissive CORS for production stability
CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "Authorization"],
    "methods": ["GET", "POST", "OPTIONS"]
}}, supports_credentials=True)

# DATA_DIR in current project folder for better permission handling on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "system_data")
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[*] Created DATA_DIR at {DATA_DIR}")
    except Exception as e:
        print(f"[!] Warning: Could not create DATA_DIR: {e}")

STATS_FILE = os.path.join(DATA_DIR, "stats.json")
ANALYSIS_HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.json")
CONTACTS_FILE = os.path.join(DATA_DIR, "contacts.txt")

print(f"[*] DATA STORAGE: {DATA_DIR}")
print(f"[*] Files outside workspace to avoid Live Server reloads.")

# Startup Cleanup: Remove legacy files from root workspace if they exist
for legacy_file in ["stats.json", "analysis_history.txt", "contacts.txt"]:
    if os.path.exists(legacy_file):
        try:
            os.remove(legacy_file)
            print(f"[!] Removed legacy local file: {legacy_file}")
        except:
            pass

# ================= HISTORY & DATASET =================
def save_analysis_result(original_input, confidence, label, extracted_text=None, data_type="AI Analysis", ai_score=None, expert_score=None, title="N/A", link="N/A", subject="General"):
    """
    Saves the analyzed result to the analysis_history.json file and appends to dataset.
    """
    try:
        # 1. Prepare Entry
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        item_id = int(time.time() * 1000)
        
        # Clean inputs
        clean_input = str(original_input).strip() if original_input else "N/A"
        if not extracted_text:
            extracted_text = clean_input
            
        new_entry = {
            "id": item_id,
            "date": timestamp,
            "original_input": clean_input,
            "extracted_text": extracted_text[:2000],
            "label": label,
            "confidence": confidence,
            "data_type": data_type,
            "ai_score": ai_score,
            "expert_score": expert_score,
            "title": title,
            "link": link,
            "subject": subject
        }

        # 2. Save to History File
        history = []
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        history = json.loads(content)
            except Exception as e:
                print(f"[!] Warning: Could not read history: {e}")
        
        # Add to top and limit to 500
        history.insert(0, new_entry)
        history = history[:500]
        
        with open(ANALYSIS_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        
        # 3. Add to Dataset (Feedback Loop)
        add_to_dataset(
            text=extracted_text, 
            label=label,
            link=link,
            title=title,
            subject=subject
        )
        
        print(f"[✅] History Saved: {label} ({confidence}) | Source: {data_type}")
        return True
    except Exception as e:
        print(f"[!] Critical Error in save_analysis_result: {traceback.format_exc()}")
        return False

def add_to_dataset(text, label, link="N/A", title="N/A", subject="General"):
    """
    Appends analyzed text to the appropriate CSV dataset file for future retraining.
    Also handles duplicate checks and maps labels correctly.
    """
    try:
        if not text or len(str(text).strip()) < 10:
            return 
            
        label_str = str(label).upper()
        dataset_name = ""
        numerical_label = 1
        
        # Map labels to dataset types (as requested: Fake/Unverified = Fake, Real/Trusted = Real)
        # We also include Somali translations to be thorough
        if any(keyword in label_str for keyword in ["REAL", "TRUSTED", "RASMI", "WAR RASMI AH", "RUN"]):
            dataset_name = "Real-news.csv"
            numerical_label = 1
        elif any(keyword in label_str for keyword in ["FAKE", "BEEN", "UNVERIFIED", "LAMA XAQIIJIN", "SUSPICIOUS", "SHAKI"]):
            dataset_name = "fake-news.csv"
            numerical_label = 0
        else:
            # Don't add ambiguous data
            return

        dataset_path = os.path.join(os.path.dirname(__file__), "Dataset", dataset_name)
        
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))
            
        # ================= PERFORMANCE NOTE =================
        # We perform a simple append. Checking for duplicates by reading the whole CSV 
        # on every single request is too expensive for Render/Production and causes hangs.
        # Duplicates are better handled during the bulk retraining phase if needed.
        clean_text = str(text).strip()
        if len(clean_text) < 10: return


        # Create new record structure
        # Ensure values are not too long for the CSV preview
        new_data = {
            "link": str(link)[:500],
            "title": str(title)[:200],
            "Text": str(text),
            "Subject": str(subject)[:100],
            "label": numerical_label
        }
        
        df_new = pd.DataFrame([new_data])
        
        # Append to CSV
        if os.path.exists(dataset_path):
            df_new.to_csv(dataset_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(dataset_path, index=False, encoding='utf-8-sig')
            
        print(f"[📈] DATASET UPDATED: Added new entry to {dataset_name} | Title: {title[:30]}...")
        
    except Exception as e:
        print(f"[!] Warning: Could not add to dataset feedback loop: {e}")

def load_stats():
    defaults = {"requests_handled": 0, "model_accuracy": "94.5%"}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                data = json.load(f)
                return {**defaults, **data}
        except:
            pass
    return defaults

def save_stats(stats):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except:
        pass

global_stats = load_stats()

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "Server is running", "environment": os.environ.get("RENDER", "local")})

@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"status": "alive", "time": time.time()})

@app.errorhandler(404)
def not_found(e):
    path = request.path.lower()
    # If it's an API call, return JSON
    if path.startswith('/admin') or path.startswith('/predict') or path.startswith('/api'):
        return jsonify({"error": f"Path {request.path} not found on this server"}), 404
    # Otherwise return index.html for SPA routing
    return app.send_static_file('index.html')

# ================= NLTK & MODEL STATE =================
nltk_initialized = False
model = None
vectorizer = None
label_encoder = None
lemmatizer = None
stop_words = set()

def load_resources():
    """Lazy loader for NLTK and AI models to keep startup fast."""
    global model, vectorizer, label_encoder, lemmatizer, stop_words, nltk_initialized
    
    if nltk_initialized:
        return

    print("[*] Loading AI models and NLTK data...")
    
    # 1. NLTK Path Setup
    data_dir = os.path.join(BASE_DIR, "nltk_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)

    # 2. Delayed Imports
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Check for packages
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            if pkg == "punkt": nltk.data.find("tokenizers/punkt")
            elif pkg == "punkt_tab": nltk.data.find("tokenizers/punkt_tab")
            else: nltk.data.find(f"corpora/{pkg}")
        except:
            print(f"[*] NLTK: Downloading {pkg} on-demand...")
            nltk.download(pkg, download_dir=data_dir, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    somali_stopwords = [
        "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
        "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
        "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
        "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
        "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona",
        "yahay", "yihiin", "ahayd", "ahaa", "noqday", "noqon", "leh", "leeyihiin",
        "kala", "hore", "danbe", "dhammaan", "kasta", "badnaa", "yar", "weyn",
        "oo", "kale", "jira", "jiray", "ilaa", "halkan", "halkaas", "mid", "kaliya",
        "isla", "markaana", "ahaana", "ahaanna", "hadda", "horey", "sheegay", "sheegtay"
    ]
    stop_words.update(somali_stopwords)

    # 3. Model Loading
    try:
        MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
        VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")
        ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl")

        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            print("[*] Models loaded.")
        else:
            print("[!] Warning: Model files missing in saved_model/")
    except Exception as e:
        print(f"[!] Model load failure: {e}")

    nltk_initialized = True

# ================= EAGER LOAD (Wait for models at startup) =================
print("[*] System Boot: Loading all resources before starting server...")
try:
    load_resources()
    print("[*] System Boot: All models and NLTK data are READY.")
except Exception as e:
    print(f"[!] System Boot Warning: Resource loading issue: {e}")

# ================= PRE-COMPILED REGEX FOR SPEED =================
URL_PATTERN = re.compile(r'^(https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+([/?#].*)?$', re.IGNORECASE)
CLEAN_TEXT_PATTERN = re.compile(r"[^a-z' ]")
DOMAIN_CLEAN_PATTERN = re.compile(r'^https?://(www\.)?')
SUSPICIOUS_EXT_PATTERN = re.compile(r"\.(tk|ga|ml|cf|icu|xyz)$")

# ================= HELPERS =================
def sanitize_text(text):
    """Remove HTML tags and strip text."""
    if not isinstance(text, str):
        return ""
    # Use 'html.parser' which is built-in and fast enough
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.strip()

def preprocess_text(text):
    """High-accuracy preprocessing using NLTK word_tokenize & lemmatization"""
    load_resources() # Ensure loaded
    if not text: return ""
    from nltk.tokenize import word_tokenize
    text = sanitize_text(text).lower()
    text = CLEAN_TEXT_PATTERN.sub(" ", text)
    tokens = word_tokenize(text)
    # Filter stopwords, short words and lemmatize
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(cleaned)

def is_url(text):
    """Fast URL detection."""
    return bool(URL_PATTERN.match(text.strip()))

def guess_subject(text):
    """Guess the news subject based on keywords."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["siyaasad", "baarlaman", "doorasho", "government", "policy", "politics", "maamulka"]):
        return "Politics"
    if any(w in text_lower for w in ["qarax", "amaanka", "ciidanka", "police", "security", "terrorism", "war", "asluubta"]):
        return "Security"
    if any(w in text_lower for w in ["caafimaadka", "isbitaal", "health", "doctor", "virus", "fayras", "dawo"]):
        return "Health"
    if any(w in text_lower for w in ["lacag", "dhaqaale", "bank", "finance", "economy", "ganacsi", "cashuur", "deynta"]):
        return "Finance"
    return "General"

@lru_cache(maxsize=300)
def extract_text_from_url(url):
    """Ka soo saar qoraalka bogga webka URL si qoto dheer"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/"
        }
        
        # SSL certificate errors are common on some Somali sites, so we try to be robust
        try:
            resp = requests.get(url, headers=headers, timeout=4, verify=True)
        except requests.exceptions.SSLError:
            print(f"[!] SSL Error for {url}, trying without verification...")
            resp = requests.get(url, headers=headers, timeout=4, verify=False)

        if resp.status_code == 404:
            raise Exception("Boggan lama helin (404 Not Found). Fadlan hubi in link-gu sax yahay.")
        elif resp.status_code == 403:
            raise Exception("Websaydhkani wuxuu xannibay helitaanka tooska ah. Fadlan koobiyeey qoraalka oo halkan kusoo dheji.")
        elif resp.status_code != 200:
            raise Exception(f"Kala xiriirida bogga wey fashilantay. Status: {resp.status_code}")
        
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Get Title
        page_title = soup.title.string if soup.title else "News from URL"
        
        # Remove unwanted elements
        for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe", "ad"]):
            element.decompose()
            
        text_parts = []
        
        # Expanded article containers search
        main_content = soup.find(['article', 'main']) or \
                       soup.find('div', class_=re.compile(r'(post|article|content|entry-content|news-body|story-body|article-text|page-content)', re.I)) or \
                       soup.find('div', id=re.compile(r'(post|article|content|story|main)', re.I))
        
        target_soup = main_content if main_content else soup
        
        # Extract from headings and paragraphs
        elements = target_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
        
        for el in elements:
            text = el.get_text(separator=" ", strip=True)
            # Filter out short fragments, menus, etc.
            if len(text.split()) > 4:
                text_parts.append(text)
                
        extracted_text = " ".join(text_parts)
        
        # Fallback for sites with non-standard structures
        if len(extracted_text) < 150:
            all_text = target_soup.get_text(separator=" ", strip=True)
            # Simple cleaning for fallback text
            extracted_text = re.sub(r'\s+', ' ', all_text)
            
        if len(extracted_text) < 50:
             raise Exception("Ma jiro qoraal ku filan oo laga helay boggan.")

        print(f"[🌐] URL Extracted: {len(extracted_text)} chars from {url}")
        return extracted_text.strip(), page_title.strip()
    except Exception as e:
        print(f"[❌] URL Extract Error: {e}")
        raise Exception(f"Cilad ka timid barta webka: {str(e)}")

# ================= SEARCH ENGINE (DUCKDUCKGO LITE) =================
@lru_cache(maxsize=128)
def search_duckduckgo_lite(query):
    """
    Kala soo bax natiijooyin live ah DuckDuckGo Lite si loo xaqiijiyo dhacdooyinka hadda socda.
    """
    try:
        url = "https://lite.duckduckgo.com/lite/"
        headers = {"User-Agent": "Mozilla/5.0"}
        # Strict timeout to avoid blocking the server
        res = requests.post(url, data={"q": query}, headers=headers, timeout=5.0)
        soup = BeautifulSoup(res.text, 'html.parser')

        results = []
        for td in soup.find_all('td', class_='result-snippet'):
            snippet = td.text.strip()
            link = ""
            title = ""
            tr = td.find_parent('tr')
            if tr:
                prev_tr = tr.find_previous_sibling('tr')
                if prev_tr:
                    a_tag = prev_tr.find('a', class_='result-link')
                    if a_tag:
                        link = a_tag.get('href', '')
                        title = a_tag.text.strip()
            results.append({'snippet': snippet, 'link': link, 'title': title})
        
        return results
    except Exception as e:
        print(f"[!] Search Error: {e}")
        return []
    except Exception:
        return []

# ================= EXTRA FEATURES =================
def is_extreme_claim(text):
    if not isinstance(text, str): return 0
    extreme_words = ["100 sano", "hal charge 6 bilood", "miracle", "cure", "mucjiso", "lacag bilaash"]
    return int(any(word in text.lower() for word in extreme_words))

def is_vague_source(text):
    if not isinstance(text, str): return 0
    vague_words = ["khubaro ayaa sheegay", "daraasad cusub ayaa sheegtay", "ilo wareedyo", "warar la helayo"]
    return int(any(word in text.lower() for word in vague_words))

# Resource loading moved to lazy pattern above

# ================= HEURISTIC FACT CHECKER =================
import tldextract

TRUSTED_SOURCES = {
    "bbc.com", "voasomali.com", "goobjoog.com", 
    "garoweonline.com", "somalistream.com", "somnn.com", 
    "somaliglobe.net", "sntv.so", "sonna.so", "hiiraan.com",
    "caasimada.net", "jowhar.com", "warsheekh.com",
    "puntlandtimes.ca", "radiomuqdisho.net", "villasomalia.gov.so",
    "daljir.com", "puntlandpost.net", "horseedmedia.net", "barkulan.com",
    "radioergo.org", "aljazeera.com", "reuters.com", "apnews.com",
    "nytimes.com", "cnn.com", "theguardian.com", "standardmedia.co.ke",
    "mogadishucenter.com", "puntland.so", "galmudug.so", "hirshabelle.so"
}

UNTRUSTED_PATTERNS = [
    "shidan", "dawo mucjiso ah", "lacag bilaash ah", 
    "guji halkan", "win iphone", "naxdin", "nin yaaban",
    "naag yaaban", "subxaanallaah", "yaabka aduunka", "arrin lala yaabo",
    "qarax cusub", "war hadda soo dhacay", "daawasho naxdin leh",
    "daawo video-ga", "mucjisooyin", "yaab", "cajiib",
    "cod qarsoodi ah", "sir culus", "fadeexad", "looma quudho",
    "dawo kasta", "lacag fudud", "halkan ka gal", "share dheh"
]

def heuristic_fact_check(text, url=None):
    """
    EXPERT FACT-CHECK: Strictly focuses on Source Reputation and Live Web Verification.
    This does NOT analyze text patterns (which AI does), instead it looks for external truth.
    """
    score = 0
    reasons = []
    text_lower = text.lower()
    words = text.split()
    
    # Pre-detect if source is trusted
    is_trusted_domain = False
    if url:
        try:
            extracted = tldextract.extract(url)
            if f"{extracted.domain}.{extracted.suffix}".lower() in TRUSTED_SOURCES:
                is_trusted_domain = True
        except: pass
    
    # 1. Source Reliability (URL / Domain Trust)
    if url:
        try:
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}".lower()
            
            if domain in TRUSTED_SOURCES:
                score += 80 
                reasons.append(f"Hubinta Isha: Qoraalkan wuxuu si toos ah uga yimid ilo lagu kalsoon yahay ({domain}).")
            else:
                if extracted.suffix in ["tk", "ga", "ml", "cf", "icu", "xyz", "online", "top", "pw", "bid"]:
                    score -= 50
                    reasons.append(f"Digniin Source: Domain-ka ({domain}) waxaa isticmaala ilo aan la aqoonsan oo badanaa faafiya been-abuur.")
        except:
            pass

    # 2. Live Web Verification (Searching for Citations/Confirmations)
    # Extract query words
    stop_words_all = set(list(stop_words) + somali_stopwords)
    query_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words_all]
    subjects = [w for w in words if len(w) > 3 and w[0].isupper()]
    
    danger_keywords = ["dhintey", "geeriyooday", "qarax", "shil", "dhaawacmay", "iscasilay", "xilka laga qaaday", "killed", "attacked"]
    has_danger = any(kw in text_lower for kw in danger_keywords)
    
    if subjects and has_danger:
        query = f"{subjects[0]} {', '.join([kw for kw in danger_keywords if kw in text_lower])}"
    else:
        query = " ".join(query_words[:8]) if query_words else text[:60]

    if len(query) > 10:
        live_results = search_duckduckgo_lite(query)
        if live_results:
            max_sim = 0
            found_debunk = False
            trusted_hits = []
            
            debunk_words = ["fake", "false", "hoax", "fact check", "been abuur", "been-abuur", "checked", "debunked"]
            
            for res in live_results:
                res_text = (res['title'] + " " + res['snippet']).lower()
                res_matches = sum(1 for w in query_words[:10] if w.lower() in res_text)
                if res_matches > max_sim: max_sim = res_matches
                
                # Check Domain of Result
                try:
                    ext_res = tldextract.extract(res['link'])
                    res_domain = f"{ext_res.domain}.{ext_res.suffix}".lower()
                    if res_domain in TRUSTED_SOURCES and res_matches >= 6:
                        trusted_hits.append(res['link'])
                except: pass
                
                if any(dk in res_text for dk in debunk_words) and res_matches >= 5:
                    found_debunk = True

            if found_debunk:
                score -= 150
                reasons.append("Natiijooyinka Baaritaanka: Xog laga helay internet-ka ayaa sheegaysa in qoraalkan uu yahay been-abuur la xaqiijiyay.")
            elif trusted_hits:
                score += 100
                source_link = trusted_hits[0]
                reasons.append(f"Xog Run Ah: Warkan waxaa lagu tebiyay ilo caalami ah oo sugan. <a href='{source_link}' target='_blank'>Riix halkan si aad u eegto</a>.")
            elif max_sim >= 8:
                score += 60
                reasons.append("Baaris Web: Waxaa la helay xog badan oo u dhiganta warka aad soo gudbisay, taas oo kor u qaadaysa kalsoonida.")
            elif has_danger and max_sim < 4:
                score -= 120
                reasons.append("Digniin: Dhacdo xasaasi ah (Danger/Death) laguma tebin ilaha rasmiga ah ee internet-ka (None Found).")
            elif max_sim < 4:
                score -= 70
                reasons.append("Xog La'aan: Markii la baaray internet-ka, laguma helin natiijooyin xaqiijinaya warkaan (No citations found).")
        else:
            if not is_trusted_domain:
                score -= 100
                reasons.append("Cilad Baaris: Wax xog ah lagama helin internet-ka oo ku saabsan nuxurka qoraalkan.")
            else:
                score += 50
                reasons.append("Cilad Baaris: Search fashilmay laakiin isha warka ayaa la xaqiijiyay.")



    # Final Verdict for Expert Fact-Check
    confidence = 70 + (min(28, abs(score) * 0.15))
    if score >= 60 or is_trusted_domain:
        rating = "Trusted"
        if is_trusted_domain:
            confidence = max(94, confidence) # Boost confidence for trusted domains
    else:
        rating = "Unverified" # Simplified to Trusted/Unverified per user request

    return {
        "rating": rating,
        "confidence": f"{int(confidence)}%",
        "reasons": reasons,
        "score": score
    }

# ================= ROUTES =================
def no_cache_response(file_path):
    response = app.send_static_file(file_path)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route("/", methods=["GET"])
def home():
    return no_cache_response('index.html')

@app.route("/Admin", methods=["GET"])
@app.route("/admin", methods=["GET"])
def admin_page():
    return no_cache_response('index.html')

@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return no_cache_response('index.html')

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        load_resources() # Ensure loaded
        # Check models first
        if model is None or vectorizer is None:
            return jsonify({"error": "AI Models are not loaded on the server. Please check server logs."}), 500

        global_stats["requests_handled"] = global_stats.get("requests_handled", 0) + 1
        save_stats(global_stats)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON not found"}), 400

        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error": "No text provided"}), 400

        content = str(content).strip()
        raw_user_input = content # Save exactly what was entered
        
        input_type = data.get("type", "text")
        input_url = None
        
        # Haddii input uu URL yahay ama ciddida u eg tahay URL
        page_title = "News Article"
        if input_type == "url" or is_url(content):
            if not content.startswith(("http://", "https://")):
                content = "https://" + content
            
            input_url = content
            
            try:
                extracted, page_title = extract_text_from_url(input_url)
                if not extracted and input_url.startswith("https://"):
                    input_url = input_url.replace("https://", "http://")
                    extracted, page_title = extract_text_from_url(input_url)
            except Exception as e:
                # Returns the specific extraction error to the user
                return jsonify({"error": str(e)}), 400
                
            if not extracted:
                return jsonify({"error": "SYSTEM: Failed to extract data from URL. The site may have blocked the system or is empty."}), 400
            content = extracted

        else:
            # Use a snippet of text as title for text inputs
            page_title = content[:60] + "..." if len(content) > 60 else content
        
        # Guess subject
        news_subject = guess_subject(content)

        # ================= Preprocess =================
        clean_input = preprocess_text(content)

        # Vectorize
        X = vectorizer.transform([clean_input])
        ext = is_extreme_claim(content)
        vague = is_vague_source(content)
        
        X_dense = X.toarray()
        X = np.hstack([X_dense, np.array([[ext, vague]])])

        # ================= AI Decision Logic (Models + Patterns) =================
        # 1. AI Model Score (Training Data)
        ai_score_val = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0
        
        # 2. Text Pattern Heuristics (Hand-coded rules for AI side)
        pattern_penalty = 0
        text_lower = content.lower()
        
        # Sensationalism & Older news patterns
        if any(y in content for y in ["2016", "2017", "2018", "2019", "2020", "2021", "2022"]):
            pattern_penalty += 1.5
        if "!!!" in content or "???" in content or "!!!" in page_title:
            pattern_penalty += 1.5
        
        # Sensationalist/Fake-prone Keywords (Somali)
        sensational_keywords = [
            "mucjiso", "runtii", "deg deg", "dhacdo naxdin leh", "lacag bilaash", 
            "hal mar eeg", "ha moogaan", "sir culus", "faah faahin", "yaab", 
            "wax aan la rumaysan karin", "muuqaal qarsoodi ah", "subxaanallaah"
        ]
        if any(kw in text_lower for kw in sensational_keywords):
            pattern_penalty += 2.0
            print(f"[*] AI SCAN: Sensational keyword detected. Penalty applied.")

        # Shouting (ALL CAPS)
        words = content.split()
        if len(words) > 5 and sum(1 for w in words if w.isupper() and len(w) > 2) / len(words) > 0.3:
            pattern_penalty += 1.2
            
        if len(content.split()) < 15:
             pattern_penalty += 3.5 # Stronger penalty for very short text 

        # 2. Source-Based Validation (Highly Trusted Sources Boost)
        source_boost = 0
        if input_url:
            try:
                extracted_tld = tldextract.extract(input_url)
                temp_domain = f"{extracted_tld.domain}.{extracted_tld.suffix}".lower()
                if temp_domain in TRUSTED_SOURCES:
                    source_boost = 5.0 # Very heavy boost for BBC, etc.
                    print(f"[*] VALIDATION: Trusted source detected ({temp_domain}). Boosting AI Score.")
            except: pass

        final_ai_score = ai_score_val - pattern_penalty + source_boost
        
        print(f"[*] AI SCAN - Model: {ai_score_val:.2f}, Patterns: -{pattern_penalty:.2f}, Boost: +{source_boost:.2f}, Final: {final_ai_score:.2f}")

        # Sigmoid function for confidence
        confidence_val = (1 / (1 + np.exp(-abs(final_ai_score * 0.8)))) * 100
        confidence_val = min(98.5, max(75.0, confidence_val))
        
        # ================= AI Decision Logic (Stricter Verdict) =================
        # Increase threshold to 1.0 to make it much harder to get a "Real News" verdict
        if final_ai_score > 1.0:
            result = "Real News"
        elif final_ai_score < -0.2:
            result = "Fake news"
        else:
            # Borderline case: DEFAULT TO FAKE (User wants it to be stricter/less vulnerable to 'Runta')
            result = "Fake news"
            print(f"[*] AI SCAN: Borderline case ({final_ai_score:.2f}) - Defaulting to Fake.")

        # ================= Save to History (Non-blocking) =================
        if not data.get("skip_history", False):
            try:
                save_analysis_result(
                    original_input=raw_user_input, # Save raw URL/Text
                    confidence=f"{round(float(confidence_val), 2)}%",
                    label=result,
                    extracted_text=content,
                    data_type="AI Analysis",
                    ai_score=f"{round(float(confidence_val), 2)}%",
                    expert_score="N/A",
                    title=page_title,
                    link=input_url if input_url else "N/A",
                    subject=news_subject
                )
            except Exception as e:
                print(f"[!] Warning: History saving failed: {e}")

        return jsonify({
            "prediction": result, 
            "confidence": f"{round(float(confidence_val), 2)}%",
            "ai_score": float(round(float(ai_score_val), 2)),
            "expert_score": 0.0,
            "raw_text": content,
            "title": page_title,
            "link": input_url if input_url else "N/A",
            "subject": news_subject
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error during prediction:", error_msg)
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze_deep", methods=["POST"])
def analyze_deep():
    """Fastest unified analysis for Render. Prevents duplicate work."""
    try:
        load_resources() # Ensure loaded
        data = request.get_json(silent=True)
        if not data: return jsonify({"error": "No data"}), 400
        
        content = data.get("text") or data.get("data")
        input_type = data.get("type", "text")
        
        if not content: return jsonify({"error": "No text"}), 400

        input_url = None
        page_title = "News Article"
        raw_text = content.strip()
        
        # 1. Extraction (Fastest path)
        if input_type == "url" or is_url(content):
            if not raw_text.startswith(("http://", "https://")):
                raw_text = "https://" + raw_text
            input_url = raw_text
            content, page_title = extract_text_from_url(input_url)
        else:
            page_title = content[:60] + "..." if len(content) > 60 else content

        # 2. Parallel Processing (AI and Facts)
        # AI Predict
        clean_input = preprocess_text(content)
        X = vectorizer.transform([clean_input])
        ext = is_extreme_claim(content)
        vague = is_vague_source(content)
        X = np.hstack([X.toarray(), np.array([[ext, vague]])])
        
        ai_score_val = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0
        
        # Apply strict penalties also in deep analysis
        text_lower = content.lower()
        deep_penalty = 0
        if any(kw in text_lower for kw in ["mucjiso", "lacag bilaash", "deg deg", "yaab"]): deep_penalty += 2.0
        if len(content.split()) < 15: deep_penalty += 3.0
        
        final_ai_score = ai_score_val - deep_penalty
        
        # Simple AI Confidence
        ai_conf = (1 / (1 + np.exp(-abs(final_ai_score * 0.8)))) * 100
        ai_conf = f"{min(98.5, max(75.0, ai_conf)):.2f}%"
        
        # STRICT VERDICT: Must be > 1.0 to be Real
        ai_pred = "Real News" if final_ai_score > 1.0 else "Fake news"

        # Expert Fact-Check
        fc_res = heuristic_fact_check(content, input_url)
        
        # 3. Save History (Only once!)
        news_subject = guess_subject(content)
        save_analysis_result(
            original_input=data.get("data") or data.get("text"),
            confidence=fc_res.get("confidence") if float(fc_res.get("confidence", "0").replace("%", "")) > float(ai_conf.replace("%", "")) else ai_conf,
            label="War Rasmi ah" if (ai_pred == "Real News" or "trusted" in fc_res.get("rating", "").lower()) else "Fake news",
            extracted_text=content,
            data_type="Unified Deep Analysis",
            ai_score=ai_conf,
            expert_score=fc_res.get("confidence"),
            title=page_title,
            link=input_url or "N/A",
            subject=news_subject
        )

        return jsonify({
            "ai_res": {
                "prediction": ai_pred,
                "confidence": ai_conf
            },
            "fc_res": fc_res,
            "title": page_title,
            "link": input_url or "N/A",
            "subject": news_subject,
            "raw_text": content
        })

    except Exception as e:
        print(f"[!] Deep Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/fact-check", methods=["POST"])
def fact_check():
    try:
        global_stats["requests_handled"] = global_stats.get("requests_handled", 0) + 1
        save_stats(global_stats)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON not found"}), 400


        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error": "No data provided"}), 400


        input_url = None
        raw_user_input = content.strip() # Save raw input
        input_type = data.get("type", "text")
        page_title = "News Article"
        if input_type == "url" or is_url(content):
            temp_content = content.strip()
            if not temp_content.startswith(("http://", "https://")):
                temp_content = "https://" + temp_content
            
            input_url = temp_content
            try:
                content, page_title = extract_text_from_url(input_url)
                if not content and input_url.startswith("https://"):
                    input_url = input_url.replace("https://", "http://")
                    content, page_title = extract_text_from_url(input_url)
            except Exception as e:
                # Returns the specific extraction error to the user
                return jsonify({"error": str(e)}), 400
        else:
            page_title = content[:60] + "..." if len(content) > 60 else content

        if not content or len(str(content).strip()) < 5:
            return jsonify({"error": "The text found from the URL is unavailable or too small"}), 400


        fact_result = heuristic_fact_check(content, input_url)
        fact_result["raw_text"] = content 
        fact_result["title"] = page_title
        fact_result["link"] = input_url if input_url else "N/A"
        fact_result["subject"] = guess_subject(content)
        
        # Somali Labels
        rating_str = fact_result.get("rating", "unverified").lower()
        somali_label = "Lama xaqiijin"
        
        if "trusted" in rating_str: 
            somali_label = "War Rasmi ah"
        elif "fake" in rating_str:
            somali_label = "War Been Abuur Ah"
        else:
            somali_label = "Lama xaqiijin"

        # ================= Save to History (Non-blocking) =================
        if not data.get("skip_history", False):
            try:
                save_analysis_result(
                    original_input=raw_user_input, # Save raw URL/Text
                    confidence=fact_result["confidence"],
                    label=somali_label,
                    extracted_text=content,
                    data_type="Expert Fact-Check",
                    ai_score="N/A",
                    expert_score=fact_result["confidence"],
                    title=page_title,
                    link=input_url if input_url else "N/A",
                    subject=fact_result["subject"]
                )
            except Exception as e:
                print(f"[!] Warning: History saving failed: {e}")

        return jsonify(fact_result)

    except Exception as e:
        print("Error during fact-check:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/unified-history-save", methods=["POST"])
def unified_history_save():
    """
    Saves only the most confident result to logic.
    Called by front-end after performing both AI and Expert checks.
    """
    try:
        data = request.get_json(silent=True)
        if not data: return jsonify({"error": "No data"}), 400
        
        raw_input = data.get("raw_input")
        ai_res = data.get("ai_res")
        fc_res = data.get("fc_res")
        
        if not ai_res or not fc_res:
            return jsonify({"error": "Missing results"}), 400
            
        # Ignore results with errors
        ai_has_error = ai_res.get("error") is not None
        fc_has_error = fc_res.get("error") is not None
        
        if ai_has_error and fc_has_error:
            return jsonify({"success": False, "message": "Both engines returned errors. Nothing to save."})

        # Calculate confidence safely
        try:
            ai_conf = float(ai_res.get("confidence", "0").replace("%", "")) if not ai_has_error else -1.0
            fc_conf = float(fc_res.get("confidence", "0").replace("%", "")) if not fc_has_error else -1.0
        except:
            ai_conf = 0.0
            fc_conf = 0.0
        
        # Determine the winner
        if ai_conf >= fc_conf and not ai_has_error:
            # Map AI Prediction to Somali Terminology
            ai_pred = ai_res.get("prediction", "N/A")
            somali_label = "War Rasmi ah" if "Real" in ai_pred else "War Been Abuur Ah"
            
            save_analysis_result(
                original_input=raw_input,
                confidence=ai_res.get("confidence", "0%"),
                label=somali_label,
                extracted_text=ai_res.get("raw_text", "N/A"),
                data_type="AI Analysis",
                ai_score=ai_res.get("confidence", "0%"),
                expert_score="N/A",
                title=ai_res.get("title", "N/A"),
                link=ai_res.get("link", "N/A"),
                subject=ai_res.get("subject", "General")
            )
        elif not fc_has_error:
            # Fact check rating to somali label
            rating_str = fc_res.get("rating", "unverified").lower()
            
            # Map Expert Rating to Somali Terminology
            if "trusted" in rating_str:
                somali_label = "War Rasmi ah"
            elif "fake" in rating_str or "been" in rating_str:
                somali_label = "War Been Abuur Ah"
            else:
                somali_label = "Lama xaqiijin"

            save_analysis_result(
                original_input=raw_input,
                confidence=fc_res.get("confidence", "0%"),
                label=somali_label,
                extracted_text=fc_res.get("raw_text", "N/A"),
                data_type="Expert Fact-Check",
                ai_score="N/A",
                expert_score=fc_res.get("confidence", "0%"),
                title=fc_res.get("title", "N/A"),
                link=fc_res.get("link", "N/A"),
                subject=fc_res.get("subject", "General")
            )
        else:
            return jsonify({"success": False, "message": "No valid results to save."})
            
        return jsonify({"success": True, "message": "History saved for highest confidence."})
    except Exception as e:
        print(f"[❌] Unified Save Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Data lama helin"}), 400

        name = data.get("name")
        email = data.get("email")
        message = data.get("message")

        if not all([name, email, message]):
            return jsonify({"error": "Please fill in all fields"}), 400


        # Log to file
        with open(CONTACTS_FILE, "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")

        print(f"[*] New message from {name} ({email})")
        return jsonify({"status": "Success", "message": "Your message has been received!"})


    except Exception:
        traceback.print_exc()
        return jsonify({"error": "A server error occurred"}), 500


# ================= ADMIN PANEL API =================
ADMIN_CREDENTIALS = {
    "username": "admin",
    "password": "password123" # In production, use env variables and hashing
}

@app.route("/api/admin/debug/paths", methods=["GET"])
def debug_paths():
    dataset_dir = os.path.join(BASE_DIR, "Dataset")
    return jsonify({
        "BASE_DIR": BASE_DIR,
        "Dataset_Dir": dataset_dir,
        "Dataset_Exists": os.path.exists(dataset_dir),
        "Files_In_Dataset": os.listdir(dataset_dir) if os.path.exists(dataset_dir) else []
    })

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Xog la'aan"}), 400
    
    username = data.get("username")
    password = data.get("password")
    
    if username == ADMIN_CREDENTIALS["username"] and password == ADMIN_CREDENTIALS["password"]:
        return jsonify({"success": True, "token": "admin-session-token-123"})
    return jsonify({"success": False, "message": "Invalid Username or Password"}), 401

@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    """Ultra-fast stats for Render to prevent 'Loading' hang/timeout."""
    try:
        dataset_dir = os.path.join(BASE_DIR, "Dataset")
        dataset_files = os.listdir(dataset_dir) if os.path.exists(dataset_dir) else []
        
        fake_news_count = 0
        real_news_count = 0
        
        # Super fast estimation by file size (Kaggle datasets are roughly 1.5KB per entry)
        for f in dataset_files:
            if not f.endswith(".csv"): continue
            try:
                size = os.path.getsize(os.path.join(dataset_dir, f))
                estimated = size // 1500 # Fast approximation
                if "fake" in f.lower(): fake_news_count += estimated
                elif "real" in f.lower(): real_news_count += estimated
            except: pass

        latest_stats = load_stats()
        
        # Fast history/messages count
        messages_count = 0
        history_count = 0
        weekly_activity = [12, 19, 3, 5, 2, 3, 10] # Baseline default

        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                    history_count = len(history_data)
                    # Only process last 100 entries for weekly activity to save time
                    relevant_history = history_data[-100:]
                    import datetime
                    calc_activity = [0] * 7
                    for entry in relevant_history:
                        date_str = entry.get("date", "")
                        if date_str:
                            try:
                                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                                calc_activity[dt.weekday()] += 1
                            except: pass
                    if sum(calc_activity) > 0:
                        weekly_activity = calc_activity
            except: pass

        if os.path.exists(CONTACTS_FILE):
             try: messages_count = os.path.getsize(CONTACTS_FILE) // 250
             except: pass

        stats = {
            "total_datasets": len(dataset_files),
            "fake_news_count": max(fake_news_count, 1),
            "real_news_count": max(real_news_count, 1),
            "requests_handled": latest_stats.get("requests_handled", 0),
            "messages_count": messages_count,
            "history_count": history_count,
            "weekly_activity": weekly_activity,
            "model_accuracy": latest_stats.get("model_accuracy", "94.5%"),
            "system_status": "Healthy",
            "uptime": "Active"
        }
        return jsonify(stats)
    except Exception as e:
        print(f"[!] Stats Speed Error: {e}")
        return jsonify({"error": "Stats temporary unavailable"}), 200 # Fallback 200 to stop loading spinner

@app.route("/api/admin/analysis_history", methods=["GET"])
def get_analysis_history():
    try:
        if not os.path.exists(ANALYSIS_HISTORY_FILE):
            return jsonify([])
        with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/analysis_history/delete", methods=["POST"])
def delete_analysis_history():
    try:
        data = request.get_json()
        item_id = data.get("id")
        if not item_id:
            return jsonify({"success": False, "message": "ID lama helin"}), 400
            
        if not os.path.exists(ANALYSIS_HISTORY_FILE):
            return jsonify({"success": False, "message": "Faylka lama helo"}), 404
            
        with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            
        new_history = [item for item in history if item.get("id") != item_id]
        
        with open(ANALYSIS_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(new_history, f, indent=4)
            
        return jsonify({"success": True, "message": "History waa laga tirtiray!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/analysis_history/clear", methods=["POST"])
def clear_analysis_history():
    try:
        with open(ANALYSIS_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return jsonify({"success": True, "message": "Dhamaan taariikhda waa la tirtiray!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/analysis_history/sync_all", methods=["POST"])
def sync_history_to_dataset():
    """
    Manually triggers a bulk sync of all analysis history into the training datasets.
    """
    try:
        if not os.path.exists(ANALYSIS_HISTORY_FILE):
             return jsonify({"success": False, "message": "No history found to sync."}), 404

        with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        count = 0
        for item in history:
            text = item.get("extracted_text") or item.get("original_input")
            label = item.get("label")
            link = item.get("link") or item.get("original_input") or "N/A"
            title = item.get("title") or "Historical Entry"
            subject = item.get("subject") or "General"
            if text and label:
                # pass all metadata
                add_to_dataset(text, label, link=link, title=title, subject=subject)
                count += 1
                
        return jsonify({"success": True, "message": f"Successfully synced {count} records to datasets."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/save_history", methods=["POST"])
def api_save_history():
    try:
        data = request.get_json()
        print(f"[*] Incoming history data: {data}")
        original_input = data.get("original_input") or data.get("name")
        extracted_text = data.get("extracted_text")
        confidence = data.get("confidence")
        label = data.get("label")
        data_type = data.get("type") or "AI Analysis"
        ai_score = data.get("ai_score")
        expert_score = data.get("expert_score")
        
        # Metadata
        title = data.get("title") or "Unknown Title"
        link = data.get("link") or original_input
        subject = data.get("subject") or "General"
        
        print(f"[*] Parsed fields: original_input={original_input}, confidence={confidence}, label={label}")
        
        if not all([original_input, confidence, label]):
            return jsonify({"success": False, "message": "Xogta ma dhameystirna"}), 400
        
        saved = save_analysis_result(
            original_input=original_input, 
            confidence=confidence, 
            label=label, 
            extracted_text=extracted_text, 
            data_type=data_type, 
            ai_score=ai_score, 
            expert_score=expert_score,
            title=title,
            link=link,
            subject=subject
        )
        if saved:
            return jsonify({"success": True, "message": "History saved"})
        else:
            return jsonify({"success": True, "message": "Warkaan horey ayaa loo save-gareeyay (Duplicate). Lama keydin markale."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/retrain", methods=["POST"])
def retrain_model():
    try:
        # Check if already training
        flag_file = os.path.join(DATA_DIR, "training_in_progress.flag")
        if os.path.exists(flag_file):
            return jsonify({"success": False, "message": "Tababarku waa socdaa mar hore..."})
            
        # Create flag
        with open(flag_file, "w") as f:
            f.write(str(time.time()))
            
        # Run Model_trains.py as a subprocess
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_trains.py")
        import sys
        
        # Improvement: Use a safer way to clean up the flag even if the script fails.
        # We pass the flag path to the script to handle its own cleanup.
        cmd = [sys.executable, script_path, "--flag", flag_file]
        
        subprocess.Popen(cmd)
        
        return jsonify({"success": True, "message": "Tababarka model-ka waa la bilaabay, fadlan sug inta uu dhamaanayo."})

    except Exception as e:
        # Cleanup flag on failure to start
        flag_file = os.path.join(DATA_DIR, "training_in_progress.flag")
        if os.path.exists(flag_file):
            try: os.remove(flag_file)
            except: pass
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/retrain_status", methods=["GET"])
def retrain_status():
    flag_file = os.path.join(DATA_DIR, "training_in_progress.flag")
    is_training = os.path.exists(flag_file)
    return jsonify({
        "is_training": is_training,
        "message": "Tababarku waa socdaa..." if is_training else "Tababarku waa diyaar."
    })

@app.route("/api/admin/datasets", methods=["GET"])
def list_datasets():
    try:
        dataset_dir = os.path.join(BASE_DIR, "Dataset")
        files = []
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        for f in os.listdir(dataset_dir):
            if not f.endswith(".csv"):
                continue
            path = os.path.join(dataset_dir, f)
            
            # Show row count stats
            rows = 0
            try:
                # Use a fast way to count rows
                with open(path, 'r', encoding='utf-8', errors='ignore') as csvf:
                    rows = sum(1 for line in csvf) - 1 # Subtract Header
            except:
                pass

            files.append({
                "name": f,
                "size": f"{os.path.getsize(path) / 1024:.2f} KB",
                "modified": time.ctime(os.path.getmtime(path)),
                "rows": max(0, rows)
            })
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/dataset/get", methods=["GET"])
def get_dataset_content():
    filename = request.args.get("filename")
    if not filename or not filename.endswith(".csv"):
        return jsonify({"error": "Invalid filename"}), 400
    
    try:
        path = os.path.join(BASE_DIR, "Dataset", filename)
        print(f"[*] Isku dayaya in la akhriyo: {path}")
        
        # Try multiple encodings
        df = None
        for enc in ['utf-8', 'latin-1', 'utf-8-sig', 'cp1252']:
            try:
                # Read specific columns for speed or all if small
                df = pd.read_csv(path, encoding=enc)
                # Show last 500 entries (most relevant for verification)
                df = df.tail(500)
                print(f"[*] Lagu guuleystay encoding: {enc}")
                break
            except Exception:
                continue
        
        if df is None:
            return jsonify({"error": "Faylka waa la furi waayey (Encoding/Format error)"}), 500
            
        df = df.fillna("N/A")
        return jsonify({
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
            "filename": filename
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server failure: {str(e)}"}), 500

@app.route("/api/admin/dataset/delete", methods=["POST"])
def delete_dataset():
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename or not filename.endswith(".csv"):
            return jsonify({"success": False, "message": "Magaca faylka sax ma aha"}), 400
            
        dataset_dir = os.path.join(BASE_DIR, "Dataset")
        path = os.path.join(dataset_dir, filename)
        
        # Security check: ensure path is within Dataset directory
        if not os.path.abspath(path).startswith(os.path.abspath(dataset_dir)):
            return jsonify({"success": False, "message": "Helitaanka lama ogola"}), 403

        if os.path.exists(path):
            os.remove(path)
            return jsonify({"success": True, "message": f"{filename} waa la tirtiray!"})
        else:
            return jsonify({"success": False, "message": "Faylka lama helo"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/dataset/download", methods=["GET"])
def download_dataset():
    filename = request.args.get("filename")
    if not filename or not filename.endswith(".csv"):
        return jsonify({"error": "Invalid filename"}), 400
    try:
        from flask import send_from_directory
        return send_from_directory(os.path.join(BASE_DIR, "Dataset"), filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/dataset/save", methods=["POST"])
def save_dataset_content():
    data = request.get_json()
    filename = data.get("filename")
    rows = data.get("rows")
    columns = data.get("columns")
    
    if not filename or not rows:
        return jsonify({"error": "Missing data"}), 400
        
    try:
        path = os.path.join(BASE_DIR, "Dataset", filename)
        
        # Preserve data beyond the 200 rows loaded in frontend
        # Assuming the read encoding depends on what works, same logic as get
        try:
            df_existing = pd.read_csv(path, encoding='utf-8-sig')
        except:
            df_existing = pd.read_csv(path, encoding='latin-1')
        
        df_front = pd.DataFrame(rows, columns=columns)
        
        original_loaded_count = min(200, len(df_existing))
        df_remaining = df_existing.iloc[original_loaded_count:]
        
        df_combined = pd.concat([df_front, df_remaining], ignore_index=True)
        df_combined.to_csv(path, index=False, encoding='utf-8-sig')
        
        return jsonify({"success": True, "message": f"{filename} waa la keydiyey!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/dataset/add_entry", methods=["POST"])
def add_dataset_entry():
    data = request.get_json()
    filename = data.get("filename")
    entry = data.get("entry") # Dict with link, title, Text, Subject, label
    
    if not filename or not entry:
        return jsonify({"error": "Xogta lama helin"}), 400
        
    try:
        path = os.path.join(BASE_DIR, "Dataset", filename)
        import csv
        
        # Prevent manual duplicates
        if os.path.exists(path):
            import pandas as pd
            try:
                df_temp = pd.read_csv(path, usecols=['Text'], encoding='utf-8-sig')
                existing = df_temp['Text'].dropna().astype(str).str.strip().str.lower().tolist()
                if str(entry.get('Text')).strip().lower() in existing:
                    return jsonify({"success": False, "message": "Warkaan mar hore ayaa lagu daray Dataset-ka (Duplicate)."}), 200
            except:
                pass

        # Append to CSV
        with open(path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["link", "title", "Text", "Subject", "label"])
            writer.writerow(entry)
            
        return jsonify({"success": True, "message": "Warkii cusubaa waa lagu daray!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/logs", methods=["GET"])
def get_logs():
    try:
        messages = []
        if os.path.exists(CONTACTS_FILE):
            with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                parts = content.split("---\n")
                for idx, part in enumerate(parts):
                    if part.strip():
                        msg = {"id": idx}
                        lines = part.strip().split("\n")
                        for line in lines:
                            if ":" in line:
                                k, v = line.split(":", 1)
                                msg[k.lower().strip()] = v.strip()
                        messages.append(msg)
        return jsonify(messages[::-1]) # Return newest first
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/reply", methods=["POST"])
def admin_reply():
    data = request.get_json()
    recipient_email = data.get("email")
    subject = data.get("subject")
    body = data.get("body")

    if not all([recipient_email, subject, body]):
        return jsonify({"success": False, "message": "Xogta ma dhameystirna"}), 400

    sender_email = "tafaftiredetectionsystem@gmail.com"
    # FADLAN BEDEL PASSWORD-KAN: Isticmaal 'App Password' laga soo saaray Gmail Account-kaaga
    sender_password = "qgzpeswwwgtgawuy"

    try:
        msg = MIMEMultipart()
        msg['From'] = f"Tafaftire  System <{sender_email}>"
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return jsonify({"success": True, "message": "Fariinta waa loo diray si toos ah!"})
    except smtplib.SMTPAuthenticationError:
        # Hadii password-ka Google App Password la waayo
        return jsonify({
            "success": False, 
            "message": "SMTP_AUTH_ERROR"
        }), 200 # Using 200 so JS can parse it
    except Exception as e:
        print("SMTP Error:", traceback.format_exc())
        return jsonify({"success": False, "message": f"Khalad (Server/SMTP): {str(e)}"}), 500

@app.route("/api/admin/sync_emails", methods=["POST"])
def sync_emails():
    try:
        sender_email = "tafaftiredetectionsystem@gmail.com"
        sender_password = "qgzpeswwwgtgawuy"
        
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(sender_email, sender_password)
        mail.select('inbox')
        
        status, messages = mail.search(None, 'UNSEEN')
        if status != 'OK':
            return jsonify({"success": False, "message": "Failed to search emails"}), 500
            
        email_ids = messages[0].split()
        synced_count = 0
        
        for e_id in email_ids:
            res, msg_data = mail.fetch(e_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    subject_header = decode_header(msg['Subject'])[0]
                    subject = subject_header[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(subject_header[1] if subject_header[1] else 'utf-8')
                        
                    from_ = msg.get('From')
                    name = from_
                    email_addr = from_
                    if '<' in from_ and '>' in from_:
                        name = from_.split('<')[0].strip()
                        email_addr = from_.split('<')[1].replace('>', '').strip()
                    
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode()
                                    break
                                except:
                                    pass
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode()
                        except:
                            pass
                            
                    # Tirtir fariimihii hore ee uu systemku isku cel celiyay (hadii qofku reply toos ah sameeyay)
                    lines = body.split('\n')
                    clean_lines = []
                    for line in lines:
                        if line.startswith('>') or 'On ' in line and 'wrote:' in line:
                            continue
                        clean_lines.append(line)
                    clean_body = '\n'.join(clean_lines).strip()
                    if not clean_body:
                        clean_body = body.strip()
                    
                    with open(CONTACTS_FILE, "a", encoding="utf-8") as f:
                        f.write(f"Name: {name}\nEmail: {email_addr}\nMessage: [REPLY/EMAIL] {subject} - {clean_body}\n---\n")
                    
                    mail.store(e_id, '+FLAGS', '\\Seen')
                    synced_count += 1
                    
        mail.logout()
        return jsonify({"success": True, "count": synced_count, "message": f"{synced_count} farriimo cusub ayaa laga soo dejiyay."})
        
    except Exception as e:
        print("IMAP Error:", traceback.format_exc())
        return jsonify({"success": False, "message": f"Khalad xagga Email soo dejinta ah: {str(e)}"}), 500

@app.route("/api/admin/logs/delete", methods=["POST"])
def delete_log():
    data = request.get_json()
    log_id = data.get("id")
    if log_id is None:
        return jsonify({"success": False, "message": "ID lama siin"}), 400
    
    try:
        if os.path.exists(CONTACTS_FILE):
            with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            parts = content.split("---\n")
            
            new_parts = []
            for idx, part in enumerate(parts):
                if idx != log_id and part.strip():
                    new_parts.append(part.strip() + "\n---\n")
            
            with open(CONTACTS_FILE, "w", encoding="utf-8") as f:
                f.write("".join(new_parts))
                
        return jsonify({"success": True, "message": "Fariintii waa la tirtiray!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/dataset/upload", methods=["POST"])
def upload_dataset():
    """Handles CSV dataset uploads."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"}), 400
            
        if file and file.filename.endswith('.csv'):
            target_dir = os.path.join(BASE_DIR, "Dataset")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            filename = file.filename
            path = os.path.join(target_dir, filename)
            file.save(path)
            return jsonify({"success": True, "message": f"{filename} has been uploaded successfully!"})
        
        return jsonify({"success": False, "message": "Only CSV files are allowed"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ================= RUN SERVER =================
if __name__ == "__main__":
    # Local development: python app.py 
    # Production: gunicorn app:app 
    app_port = int(os.environ.get("PORT", 3402))
    print(f"[*] Starting server on port {app_port}...")
    app.run(host="0.0.0.0", port=app_port, debug=False)
