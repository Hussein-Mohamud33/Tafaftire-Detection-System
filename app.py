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
import subprocess
import json
import time
from bs4 import BeautifulSoup
import pandas as pd
import smtplib
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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

def save_analysis_result(original_input, confidence, label, extracted_text=None, data_type="AI Analysis", ai_score=None, expert_score=None, title="N/A", link="N/A", subject="General"):
    try:
        # 1. Normalize input
        clean_input = str(original_input).strip() if original_input else ""
        if not clean_input:
            print("[!] ERROR: Input is empty, cannot save to history.")
            return False
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        history_file = ANALYSIS_HISTORY_FILE
        
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception as e:
                print(f"[!] Warning: Could not read history file {history_file}: {e}")
                history = []
        
        # 2. Prevent exact sequential duplicates for same type
        search_text = clean_input.lower()
        if history and len(history) > 0:
            last_entry = history[-1]
            if last_entry.get("original_input", "").lower() == search_text and last_entry.get("type") == data_type:
                print(f"[*] SKIPPING: Sequential duplicate for {data_type} detected.")
                return True # Return true because it's already "saved" from the user perspective
                
        # Create unique ID
        import random
        item_id = int(time.time() * 1000) + random.randint(1, 999)
        
        # Auto-detect link if not provided
        if link == "N/A" and clean_input.startswith("http"):
            link = clean_input
        
        new_entry = {
            "id": item_id,
            "type": data_type,
            "original_input": clean_input,
            "extracted_text": extracted_text if extracted_text else clean_input,
            "confidence": confidence,
            "label": str(label),
            "date": timestamp,
            "ai_score": ai_score,
            "expert_score": expert_score,
            "title": title,
            "link": link,
            "subject": subject
        }
        
        history.append(new_entry)
        if len(history) > 2000: history = history[-2000:]
            
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        # 3. AUTO-DATASET LINK (Feedback Loop)
        # Pass full details to add_to_dataset
        add_to_dataset(
            text=extracted_text if extracted_text else clean_input, 
            label=label,
            link=link,
            title=title,
            subject=subject
        )

        print(f"[✅] HISTORY SAVED: {data_type} | ID: {item_id} | Input: {clean_input[:30]}...")
        return True
    except Exception as e:
        print(f"[❌] ERROR SAVING HISTORY: {traceback.format_exc()}")
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
            
        # ================= DUPLICATE CHECK =================
        if os.path.exists(dataset_path):
            try:
                # Read specific columns for performance
                df_temp = pd.read_csv(dataset_path, usecols=['Text'], encoding='utf-8-sig')
                existing_texts = df_temp['Text'].dropna().astype(str).str.strip().str.lower().tolist()
                
                clean_text = str(text).strip().lower()
                if clean_text in existing_texts:
                    # Duplicate found
                    return
            except Exception as e:
                # Fallback if CSV is malformed or column missing
                print(f"[!] Warning duplicate check error: {e}")

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

# ================= NLTK SETUP =================
# Use local folder within project for NLTK packages
nltk_data_dir = os.path.join(BASE_DIR, "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

print("[*] NLTK: Initializing packages...")
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        # Check if already downloaded
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif pkg == "punkt_tab":
            nltk.data.find("tokenizers/punkt_tab")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except (LookupError, Exception):
        try:
            print(f"[*] NLTK: Downloading {pkg}...")
            nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            print(f"[!] NLTK: Failed to download {pkg}: {e}")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
somali_stopwords = [
    "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
    "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
    "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
    "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
    "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona",
    "yahay", "yihiin", "ahayd", "ahaa", "noqday", "noqon", "leh", "leeyihiin",
    "kala", "hore", "danbe", "dhammaan", "kasta", "badnaa", "yar", "weyn"
]
stop_words.update(somali_stopwords)
lemmatizer = WordNetLemmatizer()

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
    if not text: return ""
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
        res = requests.post(url, data={"q": query}, headers=headers, timeout=3.5)
        soup = BeautifulSoup(res.text, 'html.parser')

        results = []
        for td in soup.find_all('td', class_='result-snippet'):
            snippet = td.text.strip()
            link = ""
            tr = td.find_parent('tr')
            if tr:
                prev_tr = tr.find_previous_sibling('tr')
                if prev_tr:
                    a_tag = prev_tr.find('a', class_='result-link')
                    if a_tag:
                        link = a_tag.get('href', '')
            results.append({'snippet': snippet, 'link': link})
        
        return results
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

# ================= LOAD MODELS =================
try:
     
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")
    ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    print("Models loaded successfully")

except Exception as e:
    print("Model loading failed:", e)
    traceback.print_exc()
    exit(1)

# ================= HEURISTIC FACT CHECKER =================
import tldextract

TRUSTED_SOURCES = {
    "bbc.com", "voasomali.com", "goobjoog.com", 
    "garoweonline.com", "somalistream.com", "somnn.com", 
    "somaliglobe.net", "sntv.so", "sonna.so", "hiiraan.com",
    "caasimada.net", "jowhar.com", "warsheekh.com",
    "puntlandtimes.ca", "radiomuqdisho.net",
    "daljir.com", "puntlandpost.net", "horseedmedia.net",
    "radioergo.org", "aljazeera.com", "reuters.com", "apnews.com",
    "nytimes.com", "cnn.com", "theguardian.com", "standardmedia.co.ke"
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
    Analyzes news credibility based on source reputation, content patterns, 
    and live web results matching.
    """
    score = 0
    reasons = []
    
    # 1. Source Reliability (Max +60)
    if url:
        try:
            extracted = tldextract.extract(url)
            # e.g. "bbc.com" or "voasomali.com"
            domain = f"{extracted.domain}.{extracted.suffix}".lower()
            
            if domain in TRUSTED_SOURCES:
                score += 50 # Slightly reduced from 60
                reasons.append(f"Qoraalkan wuxuu ka yimid ilo lagu kalsoon yahay ({domain}).")
            else:
                reasons.append(f"Domain-ka ({domain}) kuma jiro ilaha rasmiga ah ee la yaqaan.")
                # Penalize suspicious extensions
                if extracted.suffix in ["tk", "ga", "ml", "cf", "icu", "xyz", "online", "top", "pw", "bid"]:
                    score -= 40
                    reasons.append("Cidda iska leh URL-ka ayaa looga shakisan yahay inay faafiso dacaayad.")
        except Exception:
            reasons.append("URL-ka lala xiriiray falanqaynta ma aha mid caadi ah.")


    # 2. Sensationalism & Recycled News (Max -50)
    text_lower = text.lower()
    
    # Check for old years in supposedly new news (Updated for 2026)
    old_years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    found_old_year = any(y in text for y in old_years)
    if found_old_year:
        score -= 30 
        reasons.append("Warka waxaa ku jira taariikh hore (old years) oo looga shakiyo inuu yahay mid hore loo recycled gareeyay.")

    # Check if the text is asking a question or asking to fact check
    is_prompt = any(kw in text_lower for kw in ["fact-check", "is it true", "verify this", "runtii in", "ma dhabbaa", "xaqiiqo mise", "hubi", "waa maxay"])

    # Specific danger keywords (Deep analysis)
    danger_keywords = ["dhintey", "geeriyooday", "qarax", "shil", "dhaawacmay", "iscasilay", "xilka laga qaaday", "shut down", "banned", "killed", "attacked", "la dilay", "la xiray", "dhacdo naxdin leh"]
    has_danger_keyword = any(kw in text_lower for kw in danger_keywords)

    # Somali red flags (Increased sensitivity)
    somali_red_flags = [
        "war hadda soo dhacay", "naxdin", "deg deg", "ninkii", "naagtii", 
        "mucjiso", "dawo", "lacag bilaash", "guji", "hadda daawo", 
        "subxaanallaah", "cajiib", "yaab", "fadeexad", "sir culus", 
        "cod qarsoodi ah", "daawo muuqaalka", "yaabka aduunka"
    ]
    found_red_flags = [p for p in somali_red_flags if p in text_lower]
    
    # Official / Professional keywords (Trust builders)
    official_keywords = [
        "wasaaradda", "afhayeenka", "golaha", "war-saxaafadeed", 
        "shir jaraa'id", "maamulka", "booliska", "ayaa xaqiijiyay", 
        "ayaa sheegay in", "munaasabadda", "xaafiiska", "warmurtiyeed"
    ]
    found_official = [p for p in official_keywords if p in text_lower]

    if found_red_flags:
        score -= 60 # Increased
        reasons.append(f"Qoraalka waxaa ku jira erayo kicin ah ama clickbait ({', '.join(found_red_flags[:3])}).")
    elif found_official:
        score += 5 # Reduced from 15 (Don't reward tone too much)
        reasons.append("Qoraalku wuxuu leeyahay qaab dhismeed rasmi ah oo muujinaya kalsooni.")
    elif not has_danger_keyword:
        score += 0 # Neutral tone no longer grants boost (just stays neutral)
        reasons.append("Qoraalku waa mid dhex dhexaad ah oo aan kicin lahayn.")
    else:
        score -= 30 # Increased
        reasons.append("Warkan wuxuu sheegayaa dhacdo xasaasi ah oo u baahan xaqiijin dheeraad ah.")


    # 3. Punctuation Analysis (Sensationalism)
    if "!!!" in text or "???" in text:
        score -= 25
        reasons.append("Calaamado qaylo iyo yaab ah (!!!/???) ayaa loo isticmaalay si weyn, taasoo muujinaysa kicin.")

    
    # 4. Capitalization Check (Shouting)
    words = text.split()
    if len(words) > 5:
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if (len(caps_words) / len(words)) > 0.25:
            score -= 20
            reasons.append("Qoraalku wuxuu u qoran yahay far waaweyn oo dhan (Shouting), badanaa looma adeegsado wararka saxda ah.")


    # 5. Length & Short Text Logic
    is_short = len(words) < 25
    if not is_short:
        score += 5 # Reduced from 20 (don't reward just for length)
    else:
        score -= 15 # Increased from 10
        reasons.append("Warku waa mid aad u gaaban, xog yar ayuuna xambaarsan yahay.")

    # 6. LIVE WEB SEARCH VERIFICATION (MOST IMPORTANT)
    # We only search significant words to avoid query clutter
    query_words = [w for w in words if len(w) > 3 and w.lower() not in somali_stopwords and w.lower() not in stop_words]
    
    match_count = 0
    trusted_hits = []
    
    if not is_url(text):
        query = " ".join(query_words[:12]) if query_words else text[:60]
        if len(query) > 8:
            live_results = search_duckduckgo_lite(query)
            if live_results:
                live_context_lower = " ".join([r['snippet'] for r in live_results]).lower()
                
                # Dynamic matching based on keywords
                match_count = sum(1 for w in query_words[:12] if w.lower() in live_context_lower)
                
                # Check if trusted sources reported it (Secure check)
                trusted_hits = []
                trusted_links = []
                for res_item in live_results:
                    res_snippet_low = res_item['snippet'].lower()
                    res_link = res_item['link'].lower()
                    
                    try:
                        ext_res = tldextract.extract(res_link)
                        res_domain = f"{ext_res.domain}.{ext_res.suffix}".lower()
                        if res_domain in TRUSTED_SOURCES:
                            if res_domain not in trusted_hits:
                                trusted_hits.append(res_domain)
                                trusted_links.append(res_item['link'])
                    except:
                        continue
                
                # Check for debunking keywords in search results
                debunk_keywords = ["fake", "false", "misleading", "hoax", "fact check", "been abuur", "ma dhab baa", "been abuur ah", "been-abuur", "checked", "debunked"]
                found_debunk = any(dk in live_context_lower for dk in debunk_keywords)
                
                # Logic: If fact-checking/debunking is found, it's a huge RED FLAG
                if found_debunk:
                    score -= 100 # Heavy penalty regardless of trusted hits
                    reasons.append("Internet-ka waxaa laga helay xog muujinaysa in warkan uu yahay mid la beeniyay ama laga shakisan yahay.")
                
                if trusted_hits and not found_debunk:
                    score += 85 
                    valid_link = trusted_links[0] if trusted_links else "#"
                    source_name = trusted_hits[0].split('.')[0].upper()
                    link_html = f"<a href='{valid_link}' target='_blank' style='color:#3b82f6; text-decoration:underline;'>{source_name}</a>"
                    reasons.append(f"Xogtaan waxaa si rasmi ah u xaqiijiyay ilo lagu kalsoon yahay: {link_html}.")
                elif match_count >= 10: # Increased threshold
                    score += 40
                    reasons.append(f"Warkan waxaa si isku mid ah u tebiyay ilo badan, laakiin majiraan ilo caalami ah.")
                elif is_prompt:
                    score -= 100
                    reasons.append("Sheegashooyinkan laguma helin xog ku filan oo lagu aamino.")
                else:
                    if has_danger_keyword:
                        score -= 120 # Increased
                        reasons.append("Dhacdo xasaasi ah (Dangerous Event) oo aan internet-ka laga helayn waa Red Flag weyn.")
                    elif match_count < 6: 
                        score -= 75
                        reasons.append("Baaritaan toos ah laguma helin xog dhab ah oo xaqiijinaysa warkan.")
                    else:
                        score -= 50
                        reasons.append("Xogta internet-ka laga helay kuma filna in lagu aamino xaqiiqada warkan.")
            else:
                 score -= 80 # Increased penalty
                 reasons.append("Internet-ka maba laga helin xog xitaa ka hadlaysa sheegashadan (Non-existent Claim).")

    # Determine Rating & Confidence
    confidence = 60 + (abs(score) * 0.45)
    if confidence > 98: confidence = 98

    # MUCH STRICTER THRESHOLDS
    if score >= 85 and not is_prompt: # Increased from 60 to 85
        rating = "Trusted"
    elif score < -15 or (has_danger_keyword and score <= 10) or (is_prompt and score < 10): 
        rating = "Fake Information" 
        confidence = max(88, confidence)
    elif score < 50: # Increased from 40
        rating = "Suspicious" 
        confidence = max(80, confidence)
    else:
        rating = "Unverified"
        if is_short or "Live Search laguma helin" in str(reasons):
             rating = "Suspicious" 
             confidence = max(75, confidence)
        else:
             confidence = max(60, confidence - 5)

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
        global_stats["requests_handled"] = global_stats.get("requests_handled", 0) + 1
        save_stats(global_stats)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON not found"}), 400


        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error": "No text provided"}), 400


        content = str(content).strip()

        input_type = data.get("type", "text")
        input_url = None
        
        # Haddii input uu URL yahay ama ciddida u eg tahay URL
        page_title = "News Article"
        if input_type == "url" or is_url(content):
            if not content.startswith(("http://", "https://")):
                content = "https://" + content
            
            url_to_extract = content
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

        # ================= Hybrid Decision Logic =================
        # 1. Base AI Score (LinearSVC decision function returns distance from hyperplane)
        score = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0
        
        # 2. Heuristic Check (Expert System Integration)
        h_result = heuristic_fact_check(content, input_url)
        h_score_raw = h_result.get("score", 0)
        
        # Proportional Trust Boost based on heuristic score (-100 to +150 range usually)
        # We scale it to be comparable to the SVM score which is usually -3 to +3
        trust_boost = h_score_raw / 40.0 # e.g. 80 score = +2 boost, -80 score = -2 boost
        
        # Domain Verification Extra Boost
        if input_url:
            is_verified_domain = any(t in input_url.lower() for t in TRUSTED_SOURCES)
            if is_verified_domain:
                trust_boost += 2.0
            elif h_result["rating"] == "Suspicious":
                trust_boost -= 1.0

        # Final Combined Score (Hybrid Verdict)
        # Weighting: AI (65%) + Heuristics (35%)
        # If AI is extremely positive but Heuristics is negative, AI gets suppressed
        if score > 1.0 and h_score_raw < -20:
            final_score = (score * 0.8) + (trust_boost * 1.2)
        else:
            final_score = (score * 1.4) + (trust_boost * 0.6)
        
        print(f"[*] DEEP SCAN - AI: {score:.2f}, Heuristic Boost: {trust_boost:.2f}, Raw H-Score: {h_score_raw}, Final: {final_score:.2f}")

        # Sigmoid function for confidence
        confidence_val = (1 / (1 + np.exp(-abs(final_score * 0.8)))) * 100
        confidence_val = min(98.5, max(75.0, confidence_val))
        
        # VERDICT LOGIC (MUCH STRICKTER)
        if final_score > 2.8: # Increased from 2.2
            result = "Trusted"
        elif final_score < -0.5: # Lowered from -1.0 to catch more fake info
            result = "Fake Information"
        else:
            # Neutral zone
            if score < -0.1 or h_score_raw < 5:
                result = "Fake Information"
            elif final_score > 1.2:
                result = "Suspicious" 
            else:
                # Default to suspicious if we aren't highly confident in "Trusted"
                result = "Suspicious"

        # Override for high-certainty AI
        if score < -2.5:
             result = "Fake Information"
             confidence_val = max(92, confidence_val)
        elif score > 2.5 and h_score_raw > 20:
             result = "Trusted"
             confidence_val = max(92, confidence_val)

        # ================= Save to History (Non-blocking) =================
        try:
            save_analysis_result(
                original_input=content,
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
            "ai_score": float(round(float(score), 2)),
            "expert_score": float(round(float(trust_boost), 2)),
            "raw_text": content,
            "title": page_title,
            "link": input_url if input_url else "N/A",
            "subject": news_subject
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error during prediction:", error_msg)
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
        rating_str = fact_result["rating"].lower()
        somali_label = "Lama xaqiijin"
        if "trusted" in rating_str: 
            somali_label = "War Rasmi ah"
        elif "fake" in rating_str:
            somali_label = "War Been Abuur Ah"
        elif "suspicious" in rating_str: 
            somali_label = "Shaki Baa Ku Jira"

        # ================= Save to History (Non-blocking) =================
        try:
            save_analysis_result(
                original_input=content if not input_url else input_url,
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
        return jsonify({"success": True, "token": "admin-session-token-123"}) # Simple token for demo
    return jsonify({"success": False, "message": "Invalid Username or Password"}), 401


@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    # In a real app, these would come from a DB
    # For now, we'll calculate from files
    try:
        dataset_files = os.listdir(os.path.join(BASE_DIR, "Dataset"))
        fake_news_count = 0
        real_news_count = 0
        
        for f in dataset_files:
            if "fake" in f.lower():
                with open(os.path.join(BASE_DIR, "Dataset", f), "r", encoding="utf-8", errors="ignore") as file:
                    fake_news_count += sum(1 for line in file)
            if "real" in f.lower():
                with open(os.path.join(BASE_DIR, "Dataset", f), "r", encoding="utf-8", errors="ignore") as file:
                    real_news_count += sum(1 for line in file)

        # Real performance metrics
        # Refresh the latest stats from file in case retraining finished
        latest_stats = load_stats()

        # Calculate real contact messages
        messages_count = 0
        if os.path.exists(CONTACTS_FILE):
            try:
                with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                    messages_count = sum(1 for p in content.split("---") if p.strip())
            except:
                pass
        
        # Database History Count
        history_count = 0
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                    history_count = len(history_data)
            except:
                pass

        stats = {
            "total_datasets": len(dataset_files),
            "fake_news_count": fake_news_count,
            "real_news_count": real_news_count,
            "requests_handled": latest_stats.get("requests_handled", 0),
            "messages_count": messages_count,
            "history_count": history_count,
            "model_accuracy": latest_stats.get("model_accuracy", "94.5%"),
            "system_status": "Healthy",
            "uptime": "12 days"
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/analysis_history", methods=["GET"])
def get_analysis_history():
    print("[*] Accessing analysis history database...")
    try:
        if not os.path.exists(ANALYSIS_HISTORY_FILE):
            return jsonify([])
        
        try:
            with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            history = []
            
        # Return newest first
        return jsonify(history[::-1])
    except Exception as e:
        print(f"Error getting history: {e}")
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
        # On Windows, we use shell=True sometimes but let's try direct first
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model_trains.py")
        import sys
        # Use a cross-platform command
        rm_cmd = "del" if os.name == "nt" else "rm"
        cmd = f'"{sys.executable}" "{script_path}" && {rm_cmd} "{flag_file}"'
        
        subprocess.Popen(cmd, shell=True)
        
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


# ================= RUN SERVER =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3402))
    print(f"[*] Starting server on 0.0.0.0:{port}...")
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        print(f"[!] Server failed to start: {e}")
        traceback.print_exc()
