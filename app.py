import os
import re
import traceback
import requests
import subprocess
import json
import time
import csv
import smtplib
import imaplib
import email
import nltk
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import lru_cache

# ================= FLASK INIT =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder='Front_End', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Permissive CORS for production stability
CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "Authorization"],
    "methods": ["GET", "POST", "OPTIONS"]
}}, supports_credentials=True)

# DATA_DIR synced with Model_trains.py (Home directory for persistence)
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")
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
        
        print(f"[*] History Saved: {label} ({confidence}) | Source: {data_type}")
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
        
        if any(keyword in label_str for keyword in ["REAL", "TRUSTED", "RASMI", "WAR RASMI AH", "RUN"]):
            dataset_name = "Real-news.csv"
            numerical_label = 1
        elif any(keyword in label_str for keyword in ["FAKE", "BEEN", "UNVERIFIED", "LAMA XAQIIJIN", "SUSPICIOUS", "SHAKI"]):
            dataset_name = "Fake-news.csv"
            numerical_label = 0
        else:
            # Case for Borderline/Unverified
            dataset_name = "Fake-news.csv"
            numerical_label = 0

        dataset_path = os.path.join(os.path.dirname(__file__), "Dataset", dataset_name)
        
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))
            
        # ================= PERFORMANCE NOTE =================
        # We perform a simple append. Checking for duplicates by reading the whole CSV 
        # on every single request is too expensive for Render/Production and causes hangs.
        # Duplicates are better handled during the bulk retraining phase if needed.
        clean_text = str(text).strip()
        if len(clean_text) < 10: return


        # Create new record structure matching CSV: Title, Text, Category, Label
        import csv
        file_exists = os.path.exists(dataset_path)
        
        with open(dataset_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Title", "Text", "Category", "Label"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "Title": str(title)[:200],
                "Text": str(text),
                "Category": str(subject)[:100],
                "Label": numerical_label
            })
            
        print(f"[*] DATASET UPDATED: Added new entry to {dataset_name} | Title: {title[:30]}...")
        
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

_global_stats_cache = None
def get_global_stats():
    global _global_stats_cache
    if _global_stats_cache is None:
        _global_stats_cache = load_stats()
    return _global_stats_cache

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
somali_stopwords = [
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

def load_resources():
    """Lazy loader for NLTK, NumPy and AI models to keep startup fast."""
    global model, vectorizer, label_encoder, lemmatizer, stop_words, nltk_initialized
    
    if nltk_initialized:
        return

    print("[*] Loading AI models and NLTK data...")
    import joblib
    import numpy as np
    
    # 1. NLTK Path Setup
    data_dir = os.path.join(BASE_DIR, "nltk_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)

    # 2. Delayed Imports
    # Stop on-demand downloads to prevent Render worker timeouts.
    # NLTK data should be pre-installed via build.sh
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        stop_words.update(somali_stopwords)
    except Exception as e:
        print(f"[!] NLTK resources not found. Falling back to simple processing: {e}")
        stop_words = set(somali_stopwords)

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

# ================= EAGER LOAD (Disabled for Render Startup) =================
# We now use lazy loading inside routes to prevent Gunicorn timeout.

# ================= PRE-COMPILED REGEX FOR SPEED =================
URL_PATTERN = re.compile(r'^(https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+([/?#].*)?$', re.IGNORECASE)
CLEAN_TEXT_PATTERN = re.compile(r"[^a-z' ]")
TOKEN_PATTERN = re.compile(r"\b[a-z']+\b")
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
    """High-accuracy preprocessing using regex tokenization & lemmatization (consistent with training)"""
    load_resources() # Ensure loaded
    if not text: return ""
    
    # Clean and lower
    text = sanitize_text(text).lower()
    text = CLEAN_TEXT_PATTERN.sub(" ", text)
    
    # Regex-based tokenization (Zero NLTK data dependency for tokenizing)
    tokens = TOKEN_PATTERN.findall(text)
    
    # Filter stopwords, short words and lemmatize
    cleaned = []
    for t in tokens:
        if t not in stop_words and len(t) > 2:
            if lemmatizer:
                try: cleaned.append(lemmatizer.lemmatize(t))
                except: cleaned.append(t)
            else:
                cleaned.append(t)
                
    return " ".join(cleaned)

def is_url(text):
    """Fast URL detection."""
    return bool(URL_PATTERN.match(text.strip()))

def guess_subject(text):
    """Guess the news subject based on keywords (Somali)."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["siyaasad", "baarlaman", "doorasho", "government", "policy", "maamulka", "xilka"]):
        return "Siyaasadda"
    if any(w in text_lower for w in ["qarax", "amaanka", "ciidanka", "police", "security", "dagaal", "killed", "shil"]):
        return "Amniga"
    if any(w in text_lower for w in ["caafimaadka", "isbitaal", "health", "doctor", "virus", "fayras", "dawo"]):
        return "Caafimaadka"
    if any(w in text_lower for w in ["lacag", "dhaqaale", "bank", "finance", "economy", "ganacsi", "cashuur", "deynta"]):
        return "Dhaqaalaha"
    return "Guud"

def clean_extracted_text(text):
    """
    Cleans extracted text by removing common news boilerplate, footers, 
    social media prompts, and ad-related text in Somali and English.
    """
    if not text: return ""
    
    # Common Somali and International news boilerplate patterns to remove
    boilerplate_patterns = [
        r"Your email address will not be published.*",
        r"Required fields are marked.*",
        r"Save my name, email, and website.*",
        r"Copyright © \d{4}.*",
        r"All Rights Reserved.*",
        r"Guji halkan.*",
        r"Nagala soo xiriir.*",
        r"Ku xirnow.*",
        r"Wixii faahfaahin ah.*",
        r"Read more.*",
        r"Follow us on.*",
        r"Subscribe to.*",
        r"Like us on.*",
        r"Mudug24 is an independent.*",
        r"BBC masuul kama ahan.*",
        r"Akhri xogta ku saabsan.*",
        r"Waxaa qoray:.*",
        r"Qoraalka sawirka,.*",
        r"Wararka kale ee.*",
        r"Daawo:.*",
        r"Source:.*",
        r"Lama oggola.*",
        r"Xuquuqda qoraalkan.*",
        r"Ku xayaysii.*",
        r"©\s*\d{4}.*",
        r"Contact us:.*",
        r"Email:.*",
        r"WhatsApp:.*",
        r"Facebook:.*",
        r"Twitter:.*",
        r"Telegram:.*",
        r"Instagram:.*"
    ]
    
    cleaned = text
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove multiple spaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

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
        for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "iframe", "ad", "button"]):
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
            # Skip if el is inside a footer or nav anyway (backup)
            if el.find_parent(["nav", "footer", "header"]): continue
            
            text = el.get_text(separator=" ", strip=True)
            # Filter out short fragments, menus, etc.
            if len(text.split()) > 5:
                text_parts.append(text)
                
        raw_text = " ".join(text_parts)
        extracted_text = clean_extracted_text(raw_text)
        
        # Fallback for sites with non-standard structures if too short
        if len(extracted_text) < 150:
            all_text = target_soup.get_text(separator=" ", strip=True)
            extracted_text = clean_extracted_text(re.sub(r'\s+', ' ', all_text))
            
        if len(extracted_text) < 50:
             raise Exception("Ma jiro qoraal ku filan oo laga helay boggan.")

        print(f"[URL] Extracted: {len(extracted_text)} chars from {url}")
        return extracted_text.strip(), page_title.strip()
    except Exception as e:
        print(f"[URL] Extract Error: {e}")
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

TRUSTED_SOURCES = {
    "bbc.com", "voasomali.com", "goobjoog.com", "goobjoog.net",
    "garoweonline.com", "somalistream.com", "somnn.com", 
    "somaliglobe.net", "sntv.so", "sonna.so", "hiiraan.com",
    "caasimada.net", "jowhar.com", "warsheekh.com",
    "puntlandtimes.ca", "radiomuqdisho.net", "villasomalia.gov.so",
    "daljir.com", "puntlandpost.net", "horseedmedia.net", "barkulan.com",
    "radioergo.org", "aljazeera.com", "reuters.com", "apnews.com",
    "nytimes.com", "cnn.com", "theguardian.com", "standardmedia.co.ke",
    "mogadishucenter.com", "puntland.so", "galmudug.so", "hirshabelle.so",
    "dalsanradio.com", "radiodalsan.com", "mustaqbalmedia.net", "shabellemedia.com",
    "raxanreeb.com", "keydmedia.net", "radiokulmiye.net", "universalsomalitv.net",
    "rtv.so", "daljir.com", "bbc.co.uk", "stn.so"
}

UNTRUSTED_PATTERNS = [
    "shidan", "dawo mucjiso ah", "lacag bilaash ah", "lacag bilaash",
    "guji halkan", "win iphone", "naxdin", "nin yaaban",
    "naag yaaban", "subxaanallaah", "yaabka aduunka", "arrin lala yaabo",
    "qarax cusub", "war hadda soo dhacay", "daawasho naxdin leh",
    "daawo video-ga", "mucjisooyin", "yaab", "cajiib",
    "cod qarsoodi ah", "sir culus", "fadeexad", "looma quudho",
    "dawo kasta", "lacag fudud", "halkan ka gal", "share dheh",
    "isbaaro", "si degdeg ah", "ha ka habsaamin", "lacag badan", "nasiib",
    "daawo hadda", "yaab badan", "been maaha", "runti", "dhugo", "war naxdin leh",
    "fadeexo", "ceeb", "naxdin", "ilaahayow", "ilaahow", "ilaahay", "subxanalaah",
    "nin weyn", "naag weyn", "dhacdo xanuun badan", "nin soomaali ah", "naag soomaali ah",
    "hal mar eeg", "nin naxay", "naag naxday", "mucjiso", "wax la qariyay",
    "nin weyn oo naxay", "naag weyn oo naxday", "nin yaabsaday", "naag yaabsatay",
    "ha rumaysan", "waa been", "been abuur cad", "iska jir", "war qosol leh"
]

def heuristic_fact_check(text, url=None):
    """
    EXPERT FACT-CHECK: Focuses on Source Reputation and Live Web Verification.
    Connects to online websites via real-time search and domain validation.
    """
    score = 0
    reasons = []
    text_lower = text.lower()
    words = text.split()
    
    # Expanded Trusted Sources (Somali & International)
    ADDITIONAL_TRUSTED = {"hillaac.net", "marqaannews.net", "berberatoday.com", "somalilandpost.net", "shabelle.net"}
    CURRENT_TRUSTED = TRUSTED_SOURCES.union(ADDITIONAL_TRUSTED)
    
    # 1. Source Reliability (URL / Domain Trust)
    is_trusted_domain = False
    if url:
        try:
            import tldextract
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}".lower()
            
            if domain in CURRENT_TRUSTED:
                score += 100 
                is_trusted_domain = True
                reasons.append(f"Hubinta Isha: Qoraalkan wuxuu ka yimid ilo lagu kalsoon yahay oo la xaqiijiyay ({domain}).")
            else:
                # Suspicious Extensions
                if extracted.suffix in ["tk", "ga", "ml", "cf", "icu", "xyz", "online", "top", "pw", "bid", "pw"]:
                    score -= 70
                    reasons.append(f"Digniin Source: Domain-ka ({domain}) wuxuu isticmaalayaa kordhin (. {extracted.suffix}) oo badanaa loo adeegsado degellada aan rasmiga ahayn.")
        except: pass

    # 2. Live Web Verification (Searching for Context & Citations)
    load_resources()
    
    # Improved Query Generation for better context
    stop_words_list = list(stop_words)
    meaningful_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words_list]
    
    # Use proper nouns (Capitalized) and key action verbs for most accurate search
    proper_nouns = [w for w in words if len(w) > 3 and w[0].isupper() and w.lower() not in stop_words_list]
    danger_keywords = ["dhintey", "geeriyooday", "qarax", "shil", "dhaawacmay", "iscasilay", "la xidhay", "killed", "attacked", "arrested"]
    found_danger = [kw for kw in danger_keywords if kw in text_lower]
    
    if proper_nouns and found_danger:
        search_query = f"{proper_nouns[0]} {found_danger[0]}"
    elif len(meaningful_words) >= 5:
        search_query = " ".join(meaningful_words[:6])
    else:
        search_query = text[:80]

    found_citations = False
    citation_link = None
    
    if len(search_query) > 10:
        print(f"[*] EXPERT SEARCH: Querying '{search_query}' for context...")
        live_results = search_duckduckgo_lite(search_query)
        
        if live_results:
            match_count = 0
            debunk_found = False
            
            debunk_keywords = [
                "fake", "false", "hoax", "fact check", "been abuur", "been-abuur", 
                "checked", "debunked", "been ah", "waa been", "ha rumaysan", 
                "war been ah", "beenta", "been abuur cad"
            ]
            
            for res in live_results[:8]: # Check top 8 results
                res_content = (res['title'] + " " + res['snippet']).lower()
                
                # Count word matches for relevance
                word_matches = sum(1 for w in meaningful_words[:10] if w.lower() in res_content)
                
                # Check for Debunks
                if any(dk in res_content for dk in debunk_keywords) and word_matches >= 3:
                    debunk_found = True
                    citation_link = res['link']
                    break
                
                # Check for Trusted Media Confirmation
                try:
                    ext_res = tldextract.extract(res['link'])
                    res_domain = f"{ext_res.domain}.{ext_res.suffix}".lower()
                    if res_domain in CURRENT_TRUSTED and word_matches >= 3:
                        match_count += 1
                        if not citation_link: citation_link = res['link']
                except: pass
                
                # General high-relevance match
                if word_matches >= 4:
                    match_count += 1

            if debunk_found:
                score -= 150
                reasons.append(f"Xaqiijin: Warkan waxaa horey u beeniyay ilo xog-ogaal ah. <a href='{citation_link}' target='_blank'>Eeg halkan</a>.")
            elif match_count >= 2:
                score += 120
                found_citations = True
                reasons.append(f"Verification: Warkan waxaa laga helay ilo kale oo lagu kalsoon yahay, taas oo muujinaysa inuu jiro context sugan. <a href='{citation_link}' target='_blank'>Xaqiijinta Link-ga</a>.")
            elif match_count == 1:
                score += 50
                reasons.append("Web Check: Waxaa la helay xog la xiriirta warkan, laakiin looma hayo cadaymo dhameystiran.")
            else:
                if found_danger:
                    score -= 80
                    reasons.append("Digniin: Wararka ku saabsan nabad-galyada ama geerida oo aan laga helin saxaafadda waa in laga digtoonaadaa.")
                else:
                    # If No Trusted Matches are found for a factual claim, Experts are more suspicious
                    score -= 45 
                    reasons.append("Falanqeyn Search: Ma jirto ilo xog-ogaal ah ama website-yo rasmi ah oo xaqiijinaya nuxurka qoraalkan internet-ka.")
        else:
            # Only penalize missing search results heavily if the text looks suspicious or is too short.
            # Real news often exists online, so missing results is suspicious, but we must be careful with Real News samples.
            if untrusted_matches > 0 or len(words) < 25:
                score -= 45
                reasons.append("Digniin: Ma jirto xog internet-ka laga helay oo xaqiijinaysa nuxurka qoraalkan, qoraalkuna wuxuu u muuqdaa mid shaki leh.")
            else:
                score -= 15 # Mild suspicion for unknown news
                reasons.append("Falanqeyn: Ma jirto xog rasmi ah oo laga helay internet-ka, balse qoraalka qaabkiisu waa mid hagaagsan.")

    # 3. Linguistic & Structural Patterns (Expert Layer)
    # Clickbait patterns (penalized by experts)
    cb_patterns = ["share dheh", "nasiibkaaga", "guulayso", "mucjiso", "wax la qariyay", "cod qarsoodi ah", "ha rumaysan", "waa been"]
    untrusted_matches = sum(1 for p in UNTRUSTED_PATTERNS if p in text_lower)
    
    if untrusted_matches >= 3:
        score -= (30 * untrusted_matches)
        reasons.append(f"Digniin Expert: Waxaa la helay {untrusted_matches} calaamadood oo muujinaya in qoraalku yahay mid marin habaabin ah (Clickbait/Suspicious).")
    elif untrusted_matches > 0:
        score -= 25
        reasons.append("Falanqeyn: Luuqadda qoraalka waxaa ka muuqda calaamado shaki dhalinaya.")

    # Official Terminology (Green Flags - Experts recognize formal reporting)
    official_terms = ["war-saxaafadeed", "shir jaraa'id", "wasaaradda", "hey'adda", "taliska", "ayaa sheegay", "ayaa lagu sheegay"]
    if any(ot in text_lower for ot in official_terms):
        score += 65
        reasons.append("Xog Sugan: Qoraalku wuxuu isticmaalayaa luuqad rasmi ah ama qaabka ay wararka u qoraan saxaafada xirfadleyda ah.")

    # 4. Short Text Rule (Experts are very skeptical of short, unverified claims)
    if len(words) < 25 and not found_citations and not is_trusted_domain:
        score -= 25
        reasons.append("Digniin: Qoraalku waa mid aad u gaaban, mana jirto xog dibadda ah (Citations) oo cadaynaysa.")

    # Final Confidence & Rating Logic
    confidence_val = 60 + min(39, abs(score) * 0.3)
    
    if is_trusted_domain and score > 0:
        rating = "Trusted"
        confidence_val = max(95, confidence_val)
    elif score >= 60:
        rating = "Trusted"
    elif score <= -40: # Lowered from -50 to be more decisive on Fake signals
        rating = "Fake"
    else:
        rating = "Unverified"

    return {
        "rating": rating,
        "confidence": f"{int(confidence_val)}%",
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

        gs = get_global_stats()
        gs["requests_handled"] = gs.get("requests_handled", 0) + 1
        save_stats(gs)
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
        
        import numpy as np
        X_dense = X.toarray()
        X = np.hstack([X_dense, np.array([[ext, vague]])])

        # ================= AI Decision Logic (Pure Model) =================
        # rely ONLY on the trained model as per user request
        raw_score = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0
        
        # Consistent mapping to confidence (sigmoid of absolute decision score)
        # Multiplying raw_score by a factor to normalize confidence spread
        final_ai_score = raw_score 
        
        # AI Confidence calculation (standard sigmoid-like mapping)
        import numpy as np
        ai_conf = (1 / (1 + np.exp(-abs(final_ai_score * 2.0)))) * 100
        ai_conf_str = f"{min(99.0, max(60.0, ai_conf)):.2f}%"
        
        # BALANCED VERDICT: >= 0.0 is Real
        ai_pred = "Real News" if final_ai_score >= 0.0 else "Fake news"
        
        print(f"[*] AI SCAN (Pure Model) - Raw Score: {final_ai_score:.4f}, Prediction: {ai_pred}")

        # ================= Save to History (Non-blocking) =================
        if not data.get("skip_history", False):
            try:
                save_analysis_result(
                    original_input=raw_user_input, # Save raw URL/Text
                    confidence=ai_conf_str,
                    label=ai_pred,
                    extracted_text=content,
                    data_type="AI Analysis",
                    ai_score=ai_conf_str,
                    expert_score="N/A",
                    title=page_title,
                    link=input_url if input_url else "N/A",
                    subject=news_subject
                )
            except Exception as e:
                print(f"[!] Warning: History saving failed: {e}")

        return jsonify({
            "prediction": ai_pred, 
            "confidence": ai_conf_str,
            "ai_score": float(round(float(final_ai_score), 4)),
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
        
        # Stats update
        gs = get_global_stats()
        gs["requests_handled"] = gs.get("requests_handled", 0) + 1
        save_stats(gs)

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

        import numpy as np
        clean_input = preprocess_text(content)
        X = vectorizer.transform([clean_input])
        ext = is_extreme_claim(content)
        vague = is_vague_source(content)
        X = np.hstack([X.toarray(), np.array([[ext, vague]])])
        
        raw_score = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0
        final_ai_score = raw_score 
        
        import numpy as np
        ai_conf_num = (1 / (1 + np.exp(-abs(final_ai_score * 2.0)))) * 100
        ai_conf = f"{min(99.0, max(60.0, ai_conf_num)):.2f}%"
        
        ai_pred = "Real News" if final_ai_score >= 0.0 else "Fake news"
        print(f"[*] DEEP AI SCAN (Pure Model) - Raw Score: {final_ai_score:.4f}")

        # Expert Fact-Check
        fc_res = heuristic_fact_check(content, input_url)
        
        # 3. Unified Decision: Highest Confidence Wins
        ai_conf_num = float(ai_conf.replace("%", ""))
        # Clean Expert confidence (handles bracketed values like "(98%)")
        fc_conf_str = fc_res.get("confidence", "0").replace("%", "").replace("(", "").replace(")", "")
        fc_conf_num = float(fc_conf_str)
        
        if ai_conf_num >= fc_conf_num:
            # AI is the winner
            final_label = "REAL NEWS" if ai_pred == "Real News" else "FAKE NEWS"
            winning_confidence = ai_conf
            winning_source = "Ai analysis"
        else:
            # Expert is the winner
            fc_rating = fc_res.get("rating", "Unverified").lower()
            if "trusted" in fc_rating:
                final_label = "TRUSTED"
            elif "fake" in fc_rating:
                final_label = "FAKE INFO"
            else:
                final_label = "UNVERIFIED"
            
            winning_confidence = fc_res.get("confidence")
            winning_source = "Expert Fact-check"



        # ================= Save to History (Safe Block) =================
        try:
            news_subject = guess_subject(content)
            save_analysis_result(
                original_input=data.get("data") or data.get("text"),
                confidence=winning_confidence,
                label=final_label,
                extracted_text=content,
                data_type=winning_source,
                ai_score=ai_conf,
                expert_score=fc_res.get("confidence"),
                title=page_title,
                link=input_url or "N/A",
                subject=news_subject
            )
        except Exception as save_err:
            print(f"[!] History Log Failed: {save_err}")

        return jsonify({
            "final_verdict": final_label,
            "winning_confidence": winning_confidence,
            "winning_source": winning_source,
            "ai_res": {
                "prediction": ai_pred,
                "confidence": ai_conf
            },
            "fc_res": fc_res,
            "reasons": fc_res.get("reasons", []),
            "title": page_title,
            "link": input_url or "N/A",
            "subject": news_subject,
            "status": "success"
        })

    except Exception as e:
        print(f"[!] Deep Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/fact-check", methods=["POST"])
def fact_check():
    try:
        gs = get_global_stats()
        gs["requests_handled"] = gs.get("requests_handled", 0) + 1
        save_stats(gs)
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
            somali_label = "TRUSTED"
        elif "fake" in rating_str:
            somali_label = "FAKE INFO"
        else:
            somali_label = "UNVERIFIED"

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
            somali_label = "REAL NEWS" if "Real" in ai_pred else "FAKE NEWS"
            
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
                somali_label = "TRUSTED"
            elif "fake" in rating_str or "been" in rating_str:
                somali_label = "FAKE INFO"
            else:
                somali_label = "UNVERIFIED"

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
        print(f"[!] Unified Save Error: {traceback.format_exc()}")
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

        print(f"[*] New contact message received.")
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

@app.route('/admin')
def serve_admin():
    return app.send_static_file('Admin.html')

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    """Robust stats calculation for Admin Dashboard."""
    try:
        # Absolute path detection to avoid issues with working directories
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
        
        print(f"[*] Stats Request: Checking {dataset_dir}")
        
        dataset_files = []
        if os.path.exists(dataset_dir):
            dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
        else:
            print(f"[!] Warning: Dataset directory not found at {dataset_dir}")

        fake_count = 0
        real_count = 0
        
        for f in dataset_files:
            try:
                path = os.path.join(dataset_dir, f)
                with open(path, "rb") as csv_f:
                    # Faster line counting
                    lines = sum(1 for _ in csv_f) - 1
                    if "fake" in f.lower(): fake_count += max(0, lines)
                    elif "real" in f.lower(): real_count += max(0, lines)
            except Exception as e:
                print(f"[!] Error reading dataset file {f}: {e}")

        latest_stats = load_stats()
        
        messages_count = 0
        if os.path.exists(CONTACTS_FILE):
             try:
                 with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
                     messages_count = f.read().count("---\n")
             except: pass

        history_count = 0
        weekly_activity = [0] * 7
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                import datetime
                with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                    history_count = len(history_data)
                    
                    now = datetime.datetime.now()
                    # Calculate activity for the last 7 days (Chronological order)
                    # index 0 is 6 days ago, index 6 is today
                    for entry in history_data:
                        date_str = entry.get("date", "")
                        if date_str:
                            try:
                                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                                days_diff = (now.date() - dt.date()).days
                                if 0 <= days_diff < 7:
                                    # Reverse index: today is index 6, yesterday is index 5
                                    weekly_activity[6 - days_diff] += 1
                            except: pass
            except: pass

        stats = {
            "total_datasets": len(dataset_files),
            "fake_news_count": fake_count,
            "real_news_count": real_count,
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
        print(f"[!] Stats Error: {traceback.format_exc()}")
        return jsonify({
            "total_datasets": 0,
            "model_accuracy": "94.5%",
            "system_status": "Offline",
            "error": str(e)
        })

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
        import pandas as pd
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
        import pandas as pd
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
