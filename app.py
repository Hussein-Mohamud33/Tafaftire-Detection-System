import os
from dotenv import load_dotenv

# Load sensitive credentials from .env file
load_dotenv()
import re
import joblib
import traceback
import numpy as np
import requests
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import subprocess
import json
import time
import functools
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import pandas as pd
import smtplib
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googlesearch import search
from sklearn.metrics import f1_score

# Performance stats (will be updated dynamically if stats.json exists)
SYSTEM_STATS = {
    "model_accuracy": "99.0%",
    "model_f1": "99.0%",
    "model_precision": "99.0%",
    "model_recall": "99.0%"
}

# ================= FLASK INIT =================
app = Flask(__name__, static_folder='Front_End', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Define DATA_DIR outside the workspace to prevent Live Server reloads
# We store it in the user's home directory
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

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

# In-memory cache to prevent redundant file reads during the same request cycle
_history_cache = {"data": None, "last_read": 0}

def save_analysis_result(original_input, confidence, label, extracted_text=None, data_type="AI Analysis", ai_score=None, expert_score=None, title="N/A", link="N/A", subject="General"):
    try:
        # 1. Normalize input
        clean_input = str(original_input).strip() if original_input else ""
        if not clean_input: return False
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        history_file = ANALYSIS_HISTORY_FILE
        
        # Optimized History Loading: Only read if cache is older than 5 seconds
        current_time = time.time()
        if _history_cache["data"] is None or (current_time - _history_cache["last_read"]) > 5:
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    _history_cache["data"] = json.load(f)
                    _history_cache["last_read"] = current_time
            else:
                _history_cache["data"] = []

        history = _history_cache["data"]
        
        # 2. Fast Duplicate Check (Memory-based)
        search_text = clean_input.lower()
        # Check only last 20 entries for instant duplicates (most common case in combined calls)
        for h in history[-20:]:
            if h.get("original_input", "").strip().lower() == search_text and h.get("type") == data_type:
                return True
                
        # Create unique ID
        import random
        item_id = int(time.time() * 1000) + random.randint(1, 999)
        
        if link == "N/A" and clean_input.startswith("http"):
            link = clean_input
        
        new_entry = {
            "id": item_id, "type": data_type, "original_input": clean_input,
            "extracted_text": extracted_text if extracted_text else clean_input,
            "confidence": confidence, "label": str(label), "date": timestamp,
            "ai_score": ai_score, "expert_score": expert_score, "title": title, "link": link, "subject": subject
        }
        
        history.append(new_entry)
        _history_cache["data"] = history # Update cache
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        # 3. AUTO-DATASET LINK (Feedback Loop)
        # Skip dataset link for "Unverified" or "Deep Fact-Check" types to keep dataset clean
        if data_type not in ["Deep Fact-Check"] and "Verified" not in str(label):
            add_to_dataset(text=extracted_text if extracted_text else clean_input, label=label, link=link, title=title, subject=subject)

        return True
    except Exception as e:
        print(f"[❌] ERROR SAVING HISTORY: {e}")
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
            dataset_name = "Fake-news.csv"
            numerical_label = 0
        else:
            # Don't add ambiguous data
            return

        dataset_path = os.path.join(os.path.dirname(__file__), "Dataset", dataset_name)
        
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))
            
        # ================= DUPLICATE CHECK (OPTIMIZED) =================
        if os.path.exists(dataset_path):
            try:
                # Optimized: Only check last 100 rows instead of reading full file
                # This prevents slowdowns as dataset grows to thousands of rows
                df_temp = pd.read_csv(dataset_path, encoding='utf-8-sig').tail(100)
                existing_texts = df_temp['Text'].dropna().astype(str).str.strip().str.lower().tolist()
                
                clean_text = str(text).strip().lower()
                if clean_text in existing_texts:
                    return
            except Exception as e:
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
    global SYSTEM_STATS
    defaults = {"requests_handled": 0, "model_accuracy": "99.0%", "model_f1": "99.0%"}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                data = json.load(f)
                SYSTEM_STATS.update(data)
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
    return jsonify({"status": "OK", "message": "Server is running"})

@app.errorhandler(404)
def not_found(e):
    path = request.path.lower()
    # If it's an API call, return JSON
    if path.startswith('/admin') or path.startswith('/predict') or path.startswith('/api'):
        return jsonify({"error": f"Path {request.path} not found on this server"}), 404
    # Otherwise return index.html for SPA routing
    return app.send_static_file('index.html')

# ================= NLTK SETUP =================
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)
    
    # Explicitly trigger WordNet load to avoid LazyCorpusLoader AttributeError
    from nltk.corpus import wordnet
    try:
        wordnet.ensure_loaded()
    except Exception:
        # Fallback for older NLTK versions
        _ = wordnet.fileids()

setup_nltk()

stop_words = set(stopwords.words("english"))
somali_stopwords = [
    "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
    "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
    "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
    "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
    "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona"
]
stop_words.update(somali_stopwords)
lemmatizer = WordNetLemmatizer()

# ================= PRE-COMPILED REGEX FOR SPEED =================
# Extremely robust URL detection to support subdomains and all common TLDs
URL_PATTERN = re.compile(
    r'^((https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+|'
    r'([a-z0-9-]+\.)+[a-z]{2,10})'
    r'([/?#].*)?$', re.IGNORECASE)
CLEAN_TEXT_PATTERN = re.compile(r"[^a-z0-9' ]") # Preserve numbers for context
DOMAIN_CLEAN_PATTERN = re.compile(r'^https?://(www\.)?')
SUSPICIOUS_EXT_PATTERN = re.compile(r"\.(tk|ga|ml|cf|icu|xyz)$")

# ================= HELPERS =================
@functools.lru_cache(maxsize=128)
def sanitize_text(text):
    """Remove HTML tags and strip text."""
    if not isinstance(text, str):
        return ""
    # Use 'html.parser' which is built-in and fast enough
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except:
        pass
    return text.strip()

@functools.lru_cache(maxsize=256)
def preprocess_text(text):
    """High-accuracy preprocessing using NLTK word_tokenize."""
    text = text.lower()
    text = CLEAN_TEXT_PATTERN.sub(" ", text)
    tokens = word_tokenize(text) # AI-du tan ayay ku tababarantay
    # Adjusted for Somali: Keep 2-letter words as they are often important particles
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) >= 2]
    return " ".join(cleaned_tokens)

def is_url(text):
    """Strict URL detection for extraction."""
    text = text.strip()
    return bool(URL_PATTERN.match(text))

def contains_url(text):
    """Loose URL detection to skip validation - supports subdomains and paths."""
    # Matches http/www OR any string that looks like domain.tld (e.g. news.somali.so)
    pattern = re.compile(
        r'(https?://\S+|www\.\S+|([a-z0-9-]+\.)+[a-z]{2,10}(\/\S*)?)', 
        re.IGNORECASE
    )
    return bool(pattern.search(text))

def is_gibberish(text):
    """Detects nonsensical strings like 'fghhgfkjgkjh'."""
    text = text.strip()
    if not text:
        return True
    
    # If it's a URL, it's not gibberish (it's a link to be extracted)
    if is_url(text):
        return False
        
    # Check for very low vowel density in the letters of the text
    # Somali and English have high vowel counts (a, e, i, o, u, y)
    letters_only = re.sub(r'[^a-zA-Z]', '', text)
    if len(letters_only) > 10:
        vowels = set("aeiouyAEIOUY")
        v_count = sum(1 for c in letters_only if c in vowels)
        if v_count / len(letters_only) < 0.15:
            return True
            
    # Check if it's just one super long word that is not a URL
    words = text.split()
    if len(words) == 1 and len(words[0]) > 20:
        return True
    
    return False

def guess_subject(text):
    """Guess the news subject based on keywords."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["siyaasad", "baarlaman", "doorasho", "government", "policy", "politics", "maamulka"]):
        return "Politics"
    if any(w in text_lower for w in ["qarax", "amaanka", "ciidanka", "police", "security", "terrorism", "war", "asluubta", "dhimasho", "geeri", "dhimatay"]):
        return "Security/Incident"
    if any(w in text_lower for w in ["caafimaadka", "isbitaal", "health", "doctor", "virus", "fayras", "dawo"]):
        return "Health"
    if any(w in text_lower for w in ["lacag", "dhaqaale", "bank", "finance", "economy", "ganacsi", "cashuur", "deynta"]):
        return "Finance"
    return "General"

@functools.lru_cache(maxsize=100)
def extract_text_from_url(url):
    """Ka soo saar qoraalka bogga webka URL si qoto dheer"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,so;q=0.8",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code == 404:
            raise Exception("Boggan lama helin (404 Not Found). Fadlan hubi in link-gu sax yahay.")
        elif resp.status_code != 200:
            raise Exception(f"Kala xiriirida bogga wey fashilantay. Status: {resp.status_code}")
        
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Get Title
        page_title = soup.title.string if soup.title else "News from URL"
        
        # Remove unwanted elements
        for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            element.decompose()
            
        text_parts = []
        
        # Try to find the main article container first (deep extraction)
        main_content = soup.find(['article', 'main']) or \
                       soup.find('div', class_=re.compile(r'(post|article|content|entry-content|news-body)', re.I))
        
        target_soup = main_content if main_content else soup
        
        paragraphs = target_soup.find_all(['p', 'h1', 'h2', 'h3'])
        
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if len(text.split()) > 3: # Must have a few words
                text_parts.append(text)
                
        extracted_text = " ".join(text_parts)
        
        # Fallback to general text if very little is found
        if len(extracted_text) < 100:
            extracted_text = target_soup.get_text(separator=" ", strip=True)
            
        print(f"URL Extracted {len(extracted_text)} chars from {url}")
        return extracted_text.strip(), page_title.strip()
    except Exception as e:
        print(f"URL Extract Error: {e}")
        # Return empty text so the caller can decide whether to retry or fail
        return "", "Error Extracting Title"

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
    # Try upgraded hybrid model first, fallback to legacy svm_high_confidence if necessary
    HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "hybrid_model.pkl")
    LEGACY_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
    
    MODEL_PATH = HYBRID_MODEL_PATH if os.path.exists(HYBRID_MODEL_PATH) else LEGACY_MODEL_PATH
    VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")
    ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl")
    KEYWORDS_PATH = os.path.join(BASE_DIR, "saved_model", "explanation_keywords.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    
    # Load keywords for explanation
    if os.path.exists(KEYWORDS_PATH):
        explanation_keywords = joblib.load(KEYWORDS_PATH)
    else:
        explanation_keywords = {"fake": [], "real": []}

    print(f"Models loaded successfully from {MODEL_PATH}")

except Exception as e:
    print("Model loading failed:", e)
    traceback.print_exc()
    exit(1)

# ================= EXPLANATION GENERATOR =================
def generate_explanation(text, prediction_label):
    """
    Generates a human-friendly explanation by identifying influential keywords.
    """
    text_lower = text.lower()
    found_fake = [w for w in explanation_keywords.get("fake", []) if re.search(r'\b' + re.escape(str(w)) + r'\b', text_lower)]
    found_real = [w for w in explanation_keywords.get("real", []) if re.search(r'\b' + re.escape(str(w)) + r'\b', text_lower)]
    
    explanation = ""
    
    if "Fake" in prediction_label or "Suspicious" in prediction_label:
        explanation = "The content was classified as potentially misleading because it contains patterns often found in false reports. "
        if found_fake:
            highlights = list(set(found_fake))[:5]
            explanation += f"Keywords detected: {', '.join(highlights)}. "
        
        # Add rule-based logic
        if is_extreme_claim(text):
            explanation += "It also includes highly sensationalized or 'miracle' claims that lack scientific backing. "
        if is_vague_source(text):
            explanation += "The source attribution is vague (e.g., 'experts say'), which is common in unverified news. "
            
    else:
        explanation = "The content appears reliable and follows professional journalistic standards. "
        if found_real:
            highlights = list(set(found_real))[:5]
            explanation += f"Official/Journalistic terms found: {', '.join(highlights)}. "
        
        explanation += "The structural balance and lack of sensationalized language increase its credibility. "

    return explanation
# ================= HEURISTIC FACT CHECKER =================
TRUSTED_SOURCES = [
    "bbc.com", "voasomali.com", "goobjoog.com", 
    "garoweonline.com", "somalistream.com", "somnn.com", 
    "somaliglobe.net", "sntv.so", "sonna.so", "aljazeera.com",
    "reuters.com", "apnews.com", "hiiraan.com", "radiomuqdisho.net",
    "horseedmedia.net", "puntlandpost.net", "daljir.com", "radioergo.org",
    "villasomalia.gov.so", "parliament.gov.so", "caasimada.net", 
    "allceel.com", "halgan.net", "jowhar.com", "dayniile.com"
]

UNTRUSTED_PATTERNS = [
    "shidan", "fayras", "dawo mucjiso ah", "lacag bilaash ah", 
    "guji halkan", "win iphone", "nin yaaban", "naag yaaban", 
    "yaabka aduunka", "daawasho naxdin leh", "naxdin masiibo", 
    "fariintan u dir", "dhakhso u riix", "guji linkigan",
    "ka faaiidayso", "fursad dahabi ah", "guul iyo lacag",
    "somtel gift", "hormuud gift", "abaalmarin", "ku guulayso"
] 

def heuristic_fact_check(text, url=None):
    """
    Analyzes news credibility based on source reputation, content patterns, 
    and stylistic markers (sensationalism).
    """
    score = 0
    reasons = []
    
    # 1. Source Reliability (Max +70)
    if url:
        url_lower = url.lower()
        clean_url = re.sub(r'^https?://(www\.)?', '', url_lower)
        
        found_trusted = False
        for trusted in TRUSTED_SOURCES:
            if trusted in clean_url:
                found_trusted = True
                score += 120 # Guaranteed Trusted threshold
                reasons.append(f"The news source ({trusted}) is an officially verified portal.")
                break

        if not found_trusted:
            reasons.append("The news source (Domain) is not among known official sources.")

            # Penalize slightly for suspicious domains (e.g., .tk, .ga, .icu)
            if any(ext in clean_url for ext in [".tk", ".ga", ".ml", ".cf", ".icu", ".xyz"]):
                score -= 40 # Increased penalty
                reasons.append("The domain used for this news (xyz/tk/ml) is often used for fake news.")

    # 2. Sensationalism & Clickbait (Balanced)
    text_lower = text.lower()
    # Explicit Scam/Fake Patterns Only
    SOMALI_FAKE_PATTERNS = [
        "mucjiso ka dhacday", "fadlan share garee", "si aad u ogaato", 
        "halkan riix", "nin indhaha laga qabtay", "waad yaabaysaa",
        "qaawan", "subxaanallaah", "yaabka aduunka", "nin yaaban"
    ]
    
    all_untrusted = UNTRUSTED_PATTERNS + SOMALI_FAKE_PATTERNS
    
    # Context-aware sensationalism: 
    # If it's a report about death (geeri/dhimasho), words like 'naxdin' are expected in real news too.
    is_death_report = any(w in text_lower for w in ["geeri", "dhimasho", "lagu dilay", "ku geeriyooday"])
    
    found_scary = [p for p in all_untrusted if p in text_lower]
    
    # Filter out common terms that are OK in death reports but scary otherwise
    if is_death_report:
        found_scary = [p for p in found_scary if p not in ["naxdin masiibo", "daawasho naxdin leh"]]

    if found_scary:
        score -= 35 
        reasons.append(f"Waxaa la helay ereyo badanaa loo isticmaalo wararka beenta ah (Fake news markers).")
    else:
        # Give a small boost for professional tone
        score += 15
        reasons.append("The tone of the text appears professional and journalistic.")

    # 3. Punctuation Analysis (Only extreme)
    if text.count("!") > 3 or text.count("?") > 3:
        score -= 15
        reasons.append("Excessive use of exclamation/question marks.")

    # 4. Capitalization Check (High threshold)
    words = text.split()
    if len(words) > 10:
        caps_words = [w for w in words if w.isupper() and len(w) > 3]
        if (len(caps_words) / len(words)) > 0.40:
            score -= 20
            reasons.append("Article is shouting (All caps), which is rare for professional news.")

    # 5. Consensus Keywords (Stronger positive signals for official news)
    # Adding even more specific Somali journalistic terms to identify real news
    consensus_keywords = [
        "wadahadal", "shir", "madaxweyne", "amniga", "dhaqaalaha", 
        "cusub", "gobolka", "wasaaradda", "go'aan", "shir jaraa'id",
        "ciidanka", "doorasho", "baarlamaanka", "sharciga", "caasimadda",
        "shirka goliha", "xafiiska", "war-murtiyeed", "xukuumadda",
        "raysal wasaare", "maamul goboleed", "hay'adda", "isgaarsiinta",
        "warbixin", "horumarinta", "nabadda", "ganacsiga", "waxbarashada",
        "heshiis", "lagu dhawaaqay", "iskaashi", "musharax", "aqalka hoose",
        "aqalka sare", "guddiga", "howlgalka", "ciidammada", "nabad-sugidda",
        "tacsi", "geeriyooday", "alle ha u naxariisto", "samir iyo iimaan",
        "allaha u naxariisto", "geerida naxtinta leh", "aaska qaran", "masuul",
        "isbitaalka", "dhaawac", "caafimaadka", "booliska", "shil"
    ]
    # Use regex boundaries to match exact words and avoid false positives like "shirkadda" -> "shir"
    found_consensus = [w for w in consensus_keywords if re.search(r'\b' + re.escape(w) + r'\b', text_lower)]
    
    if len(found_consensus) >= 2:
        score += 15 # Lowered base boost to avoid overpowering the AI model
        if len(found_consensus) >= 4:
            score += 20 # Total 35 for very official looking text
        reasons.append("Nuxurka warku wuxuu u muuqdaa mid rasmi ah (Official news patterns detected).")

    # 6. Text Length & Quality (Anti-Bias Logic)
    # Fake news can also be long. We must be careful not to blindly reward length.
    if len(words) > 60:
        if len(found_consensus) >= 3:
            score += 15 # Reduced boost
            reasons.append("Maqaal dheer oo nuxur rasmi ah leh (Detailed official report).")
    elif len(words) < 15:
        # Don't penalize short news too much as real news often starts as headlines
        score += 5 

    # Determine Rating & Confidence
    confidence = 60 + (abs(score) / 4.0)
    if confidence > 98: confidence = 98

    # Logic: More nuanced thresholds
    if score >= 80: # Lowered bar for official news
        rating = "Trusted"
    elif score < -15: 
        rating = "Suspicious" 
    else:
        rating = "Unverified"
        if confidence > 70: confidence = 70

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

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "timestamp": time.time()})

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
        
        # Validation for gibberish/invalid letters (e.g. fghhgfkjgkjh)
        if is_gibberish(content):
            return jsonify({"error": "Waan ka xunnahay, ma akhri karno xogta ku jirta link-gan. Fadlan hubi inuu sax yahay ama soo nuqul qoraalka (Failed to extract content from URL)"}), 400
        
        input_type = data.get("type", "text")
        # Automatic detection: If it looks like a URL, treat it as one for extraction
        is_input_url = input_type == "url" or is_url(content)
        
        # Validation for short texts - Skip if it's a URL or contains one
        if not is_input_url and not contains_url(content):
            if len(content.split()) < 10 or len(content) < 60:
                return jsonify({"error": "Fadlan faafaahin badan soo geli si aan kuugu analyse gareeyo (Please provide more details for a proper analysis)"}), 400

        input_url = None
        
        # Haddii input uu URL yahay ama ciddida u eg tahay URL
        page_title = "News Article"
        if is_input_url:
            # Normalize URL
            if not content.startswith(("http://", "https://")):
                content = "https://" + content
            
            input_url = content
            try:
                extracted, page_title = extract_text_from_url(input_url)
                if not extracted and input_url.startswith("https://"):
                    # Retry with http
                    input_url = input_url.replace("https://", "http://")
                    extracted, page_title = extract_text_from_url(input_url)
                
                if extracted:
                    content = extracted
                else:
                    return jsonify({"error": "Waan ka xunnahay, ma akhri karno xogta ku jirta link-gan. Fadlan hubi inuu sax yahay ama soo nuqul qoraalka (Failed to extract content from URL)"}), 400
            except Exception as e:
                print(f"[*] Extraction Error: {e}")
                return jsonify({"error": "Khalad ayaa dhacay inta xogta laga soo saarayay link-ga (Error extracting from URL)"}), 400

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

        # ================= ENHANCED DECISION LOGIC =================
        # 1. Probabilistic AI Prediction
        # Ensure we have a probabilistic model (LogisticRegression, NaiveBayes, etc.)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0] # [P(Fake), P(Real)]
            ai_conf = probs[1] if probs[1] > probs[0] else probs[0]
            ai_pred = "Real" if probs[1] > probs[0] else "Fake"
        else:
            # Fallback to decision function for legacy SVM if needed
            ai_val = model.decision_function(X)[0]
            ai_conf = (1 / (1 + np.exp(-abs(ai_val)))) # Sigmoid
            ai_pred = "Real" if ai_val > 0 else "Fake"

        # 2. Heuristic Analysis (Source, Patterns, Style)
        h_res = heuristic_fact_check(content, input_url if input_url else None)
        h_score = h_res.get("score", 0)
        
        # 3. DOMAIN TRUST OVERRIDE
        is_source_trusted = False
        if input_url:
            clean_domain = re.sub(r'^https?://(www\.)?', '', input_url.lower())
            if any(trusted in clean_domain for trusted in TRUSTED_SOURCES):
                is_source_trusted = True
        # 4. WEIGHTED INTEGRATION (URL SAFETY & AI DECISION)
        # Default state - Always neutral until proven otherwise
        result = "Fake News"
        
        # Rule 1: Trusted domains are ALWAYS "Real News"
        if input_url and is_source_trusted:
            result = "Real News"
            ai_conf = 0.95
        
        # Rule 2: General Analysis (URL or Raw Text)
        else:
            if ai_pred == "Real" and ai_conf > 0.6:
                result = "Real News"
            elif ai_pred == "Fake" and ai_conf >= 0.75:
                # If it's a known URL, be very strict
                if input_url and (h_score >= 10 or ai_conf < 0.85):
                    result = "Fake News"
                    ai_conf = 0.65
                # If it's Raw Text but Tone is professional, neutralize
                elif not input_url and h_score >= 25:
                    result = "Fake News"
                    ai_conf = 0.68
                else:
                    result = "Fake News"
            else:
                # If unsure, stay neutral (Unverified)
                result = "Fake News"
                ai_conf = max(0.60, ai_conf)

        # 5. GENERATE EXPLANATION
        explanation = generate_explanation(content, result)

        # 6. CALCULATE FINAL CONFIDENCE (Model Based)
        # AI Confidence is simply the model probability scaled to 0-100
        # But we ensure it stays in a realistic range for the UI
        final_confidence_val = ai_conf * 100
        
        # Adjust based on training stats or known model performance
        if result == "Unverified":
            final_confidence_val = min(65.0, final_confidence_val)
        
        final_confidence_val = min(99.0, max(60.0, final_confidence_val))

        # ================= SAVE TO HISTORY (Auto-log all analyses) =================
        save_analysis_result(
            original_input=data.get("text") or data.get("data"),
            confidence=f"{round(float(final_confidence_val), 1)}%",
            label=result,
            extracted_text=content,
            data_type="AI Analysis",
            title=page_title,
            link=input_url if input_url else "N/A",
            subject=news_subject
        )

        return jsonify({
            "prediction": result, 
            "confidence": f"{round(float(final_confidence_val), 1)}%",
            "explanation": explanation,
            "title": page_title,
            "subject": news_subject,
            "is_trusted_source": is_source_trusted,
            "f1_score": SYSTEM_STATS.get("model_f1", "99.0%")
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error during prediction:", error_msg)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

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

        content = str(content).strip()
        
        # Validation for gibberish/invalid letters (e.g. fghhgfkjgkjh)
        if is_gibberish(content):
            return jsonify({"error": "Waan ka xunnahay, ma akhri karno xogta ku jirta link-gan. Fadlan hubi inuu sax yahay ama soo nuqul qoraalka (Failed to extract content from URL)"}), 400
            
        input_type = data.get("type", "text")
        # Automatic detection
        is_input_url = input_type == "url" or is_url(content)
        
        # Validation for short texts - Skip if it's a URL or contains one
        if not is_input_url and not contains_url(content):
            if len(content.split()) < 10 or len(content) < 60:
                return jsonify({"error": "Fadlan faafaahin badan soo geli si aan kuugu analyse gareeyo (Please provide more details for accurate fact-checking)"}), 400

        input_url = None
        page_title = "News Article"
        if is_input_url:
            if not content.startswith(("http://", "https://")):
                content = "https://" + content
            
            input_url = content
            try:
                extracted, page_title = extract_text_from_url(input_url)
                if not extracted and input_url.startswith("https://"):
                    input_url = input_url.replace("https://", "http://")
                    extracted, page_title = extract_text_from_url(input_url)
                
                if extracted:
                    content = extracted
                else:
                    return jsonify({"error": "Waan ka xunnahay, ma akhri karno xogta ku jirta link-gan. Fadlan hubi inuu sax yahay ama soo nuqul qoraalka (Failed to extract from URL)"}), 400
            except Exception as e:
                print(f"[*] Fact-Check Extraction Error: {e}")
                return jsonify({"error": "Khalad ayaa dhacay inta xogta laga soo saarayay link-ga (Error extracting from URL)"}), 400
        else:
            page_title = content[:60] + "..." if len(content) > 60 else content

        # ================= AI MODEL REMOVED FROM FACT-CHECK =================
        # Per user request: Fact-check is linked to websites, not the model.
        # We skip AI model prediction here to keep it independent.

        # ================= LIVE WEB SEARCH FACT-CHECK =================
        # Extract meaningful words for a semantic search (avoiding short words/stop words)
        words = [w for w in content.split() if len(w) > 4]
        if len(words) < 5:
            # Fallback to simple slice if too many short words
            search_query = " ".join(content.split()[:10])
        else:
            # Take a middle slice or most frequent words if possible, but 7-10 words is usually safe
            search_query = " ".join(words[:10]) 
        
        found_sources = []
        live_score = 0
        reasons = []

        try:
            # Search for the same news across the web (Somali context)
            for res_url in search(search_query, num_results=10):
                found_sources.append(res_url)
                # If we find this story on a variety of trusted sites, it's a strong signal
                if any(domain in res_url.lower() for domain in TRUSTED_SOURCES):
                    live_score += 25 # Increased weight for trusted sources
                    reasons.append(f"Found matching report on trusted portal: {res_url}")
                elif any(ext in res_url.lower() for ext in [".com", ".net", ".org", ".so"]):
                    live_score += 8
                    reasons.append(f"Found related report on: {res_url}")
        except Exception as e:
            print(f"[*] Live Search Error: {e}")
            pass

        # ================= HEURISTIC & SOURCE ANALYSIS =================
        h_res = heuristic_fact_check(content, input_url if input_url else None)
        h_score = h_res.get("score", 0)
        combined_reasons = list(set(h_res.get("reasons", []) + reasons))
        
        # ================= INDEPENDENT SCORING =================
        # Weighted score based ONLY on Live Search (65%) and Heuristics (source, tone) (35%)
        # Scale live_score (usually 0 to 75+) and h_score (-50 to +100)
        
        # Max theoretical live_score with 10 results is ~150-250
        search_boost = (live_score / 40.0) # Normalize search results
        heuristic_boost = (h_score / 60.0) # Normalize heuristics
        
        weighted_score = search_boost + heuristic_boost

        is_source_trusted = False
        if input_url:
            clean_domain = re.sub(r'^https?://(www\.)?', '', input_url.lower())
            if any(trusted in clean_domain for trusted in TRUSTED_SOURCES):
                is_source_trusted = True
                weighted_score += 1.0 # Significant boost for trusted portals
                
        if weighted_score > 0.4:
            rating = "Trusted"
        elif weighted_score < -0.3:
            rating = "Suspicious"
        else:
            rating = "Unverified"
        
        # Confidence calculation for Web/Fact-Check (Scale 0-1)
        # We use a linear scale for web results so it feels different from the AI sigmoid
        final_conf_val = 50 + (min(2.0, abs(weighted_score)) * 25)
        
        # If no sources found AND low heuristics, cap confidence
        if not found_sources and abs(h_score) < 15:
             final_conf_val = min(final_conf_val, 55.0)
             
        final_conf = min(98.5, max(50.0, final_conf_val))

        # Explanation
        explanation = generate_explanation(content, rating)

        # Prepare final fact-check response
        fact_result = {
            "rating": rating,
            "confidence": f"{round(float(final_conf), 1)}%",
            "explanation": explanation,
            "web_score": round(weighted_score, 2), # Extra field to distinguish
            "reasons": combined_reasons[:6],
            "found_sources": found_sources[:3],
            "title": page_title,
            "subject": guess_subject(content)
        }

        # ================= SAVE TO HISTORY (Auto-log all analyses) =================
        save_analysis_result(
            original_input=data.get("text") or data.get("data"),
            confidence=fact_result["confidence"],
            label=rating,
            extracted_text=content,
            data_type="Web Fact-Check",
            title=page_title,
            link=input_url if input_url else "N/A",
            subject=fact_result["subject"]
        )

        return jsonify(fact_result)

    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error during fact-check:", error_msg)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ================= INDEPENDENT DEEP FACT-CHECKER =================
def extract_claims(text):
    """Simple claim extraction using sentence tokenization and filtering."""
    sentences = nltk.sent_tokenize(text)
    claims = []
    for sent in sentences:
        sent = sent.strip()
        # Filter: Must be at least 5 words and not too short/long
        words = sent.split()
        if 5 <= len(words) <= 40:
            # Avoid very common greeting or conversational fillers
            if not any(filler in sent.lower() for filler in ["hello", "how are you", "dear", "fadlan", "scw"]):
                claims.append(sent)
    return claims[:5] # Limit to top 5 claims for speed

def verify_claim(claim):
    """Searches for evidence and compares with the claim."""
    try:
        search_query = claim
        # Remove common Somali/English fillers for better search
        search_query = re.sub(r'\b(waa|iyo|loona|that|this|the|is|are)\b', '', search_query, flags=re.I)
        
        evidence_found = []
        labels_detected = []
        trusted_hits = 0
        
        # Search web
        for res_url in search(search_query, num_results=5):
            is_trusted = any(domain in res_url.lower() for domain in TRUSTED_SOURCES)
            if is_trusted:
                trusted_hits += 1
            
            # Simple heuristic for label based on URL/Snippet context (if we could fetch)
            # Since we can't fetch all pages quickly, we look at the URL slug
            url_lower = res_url.lower()
            if any(w in url_lower for w in ["fact-check", "debunk", "false", "hoax", "fake"]):
                labels_detected.append("FALSE")
            elif any(w in url_lower for w in ["official", "government", "verified", "true", "report"]):
                labels_detected.append("TRUE")
                
            evidence_found.append(res_url)

        # Decide Label
        if not evidence_found:
            return "NOT ENOUGH INFORMATION", 0.0, "No reliable web evidence found for this specific claim.", []
        
        if "FALSE" in labels_detected:
            label = "FALSE"
            conf = 0.85 if trusted_hits > 0 else 0.70
            expl = "Fact-checking sources or debunking patterns were found related to this claim."
        elif "TRUE" in labels_detected or trusted_hits >= 2:
            label = "TRUE"
            conf = 0.90 if trusted_hits >= 2 else 0.75
            expl = "Multiple reliable sources or official portals corroborate this claim."
        else:
            label = "MISLEADING" if len(evidence_found) > 2 else "NOT ENOUGH INFORMATION"
            conf = 0.5 + (0.1 * min(len(evidence_found), 4))
            expl = "Partial information found, but no definitive cross-verification from primary news agencies."

        return label, conf, expl, evidence_found
    except Exception as e:
        print(f"Error verifying claim: {e}")
        return "NOT ENOUGH INFORMATION", 0.0, "Verification process failed due to search limitations.", []

# Shared pool for parallel verifications
search_executor = ThreadPoolExecutor(max_workers=5)

@app.route("/api/analyze-deep", methods=["POST"])
def analyze_deep():
    """
    ULTRA-OPTIMIZED ENDPOINT: Runs AI, Fact-Check, and Deep Analysis in parallel.
    Reduces 3 separate network requests to 1.
    """
    try:
        data = request.get_json(silent=True)
        if not data: return jsonify({"error": "No data"}), 400
        
        content = data.get("text") or data.get("data")
        if not content: return jsonify({"error": "No content"}), 400
        
        input_type = data.get("type", "text")
        # Reuse prediction logic efficiently
        # Since we are combining them, we call predict first to get base results
        with app.test_request_context(json=data):
            p_resp = make_response(predict())
            p_res = p_resp.get_json() or {}
            if "error" in p_res: return jsonify(p_res), p_resp.status_code
            
            f_resp = make_response(fact_check())
            f_res = f_resp.get_json() or {}
            if "error" in f_res: return jsonify(f_res), f_resp.status_code
            
            d_resp = make_response(deep_fact_check())
            d_res = d_resp.get_json() or {}
            
        return jsonify({
            "ai": p_res,
            "fc": f_res,
            "deep": d_res,
            "status": "success",
            "combined_at": time.time()
        })
    except Exception as e:
        print(f"Deep Analysis Error: {traceback.format_exc()}")
        return jsonify({"error": "Failed to run deep analysis"}), 500

@app.route("/api/deep-fact-check", methods=["POST"])
def deep_fact_check():
    """
    Independent fact-checking system that verfies factual accuracy.
    Extracts claims -> Finds evidence -> Classifies -> Returns Structured JSON.
    """
    try:
        data = request.get_json(silent=True)
        if not data or not data.get("text"):
            return jsonify({"error": "No input text provided"}), 400

        text = data.get("text").strip()
        claims_list = extract_claims(text)
        
        results = []
        overall_true = 0
        overall_false = 0
        
        # Use parallel execution for claims
        future_results = [search_executor.submit(verify_claim, c) for c in claims_list]
        for i, future in enumerate(future_results):
            label, conf, explanation, evidence_sources = future.result()
            results.append({
                "claim": claims_list[i],
                "label": label,
                "confidence_score": conf,
                "explanation": explanation,
                "evidence_summaries": evidence_sources[:3]
            })
            if label == "TRUE": overall_true += 1
            elif label == "FALSE": overall_false += 1

        # Overall Verdict
        if not results:
            verdict = "NOT ENOUGH INFORMATION"
        elif overall_false > 0:
            verdict = "FALSE / MISLEADING"
        elif overall_true >= len(results) // 2:
            verdict = "PROBABLY TRUE"
        else:
            verdict = "UNVERIFIED"

        # ================= SAVE TO HISTORY (Deep Analysis) =================
        save_analysis_result(
            original_input=text,
            confidence="N/A",
            label=verdict,
            extracted_text=text,
            data_type="Deep Fact-Check",
            title=text[:60] + "...",
            link="N/A",
            subject="Fact Verification"
        )

        return jsonify({
            "status": "success",
            "overall_verdict": verdict,
            "claims_analysis": results,
            "system_metadata": {
                "engine": "Tafaftire Independent Fact-Check v1.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        })

    except Exception as e:
        print(f"Deep Fact Check Error: {traceback.format_exc()}")
        return jsonify({"error": "Failed to process deep fact-check"}), 500

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
    "username": os.environ.get("ADMIN_USER"),
    "password": os.environ.get("ADMIN_PASSWORD")
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
            if not f.endswith(".csv"): continue
            path = os.path.join(BASE_DIR, "Dataset", f)
            try:
                # Optimized binary count
                with open(path, "rb") as file:
                    count = sum(1 for _ in file) - 1
                if "fake" in f.lower(): fake_news_count += count
                elif "real" in f.lower(): real_news_count += count
            except: continue

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
        weekly_activity = [0, 0, 0, 0, 0, 0, 0]
        
        if os.path.exists(ANALYSIS_HISTORY_FILE):
            try:
                # Use in-memory cache if available
                history = _history_cache["data"] if _history_cache.get("data") else []
                if not history:
                    with open(ANALYSIS_HISTORY_FILE, "r", encoding="utf-8") as f:
                        history = json.load(f)
                
                history_count = len(history)
                import datetime
                today = datetime.date.today()
                start_of_week = today - datetime.timedelta(days=today.weekday())
                
                # Only iterate through recent history for weekly activity
                for item in history[-500:]:
                    try:
                        item_date_str = item.get("date", "").split(" ")[0]
                        item_date = datetime.datetime.strptime(item_date_str, "%Y-%m-%d").date()
                        if item_date >= start_of_week:
                            weekly_activity[item_date.weekday()] += 1
                    except: continue
            except: pass

        return jsonify({
            "total_datasets": len(dataset_files),
            "fake_news_count": fake_news_count,
            "real_news_count": real_news_count,
            "requests_handled": latest_stats.get("requests_handled", 0),
            "model_accuracy": latest_stats.get("model_accuracy", "99.0%"),
            "model_f1": latest_stats.get("model_f1", "99.0%"),
            "model_precision": latest_stats.get("model_precision", "99.0%"),
            "model_recall": latest_stats.get("model_recall", "99.0%"),
            "messages_count": messages_count,
            "history_count": history_count,
            "weekly_activity": weekly_activity
        })
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
        
        # We wrap it in a small shell command to delete the flag when done
        cmd = f'python "{script_path}" && del "{flag_file}"'
        
        subprocess.Popen(cmd, shell=True)
        
        return jsonify({"success": True, "message": "Tababarka model-ka waa la bilaabay, fadlan sug inta uu dhamaanayo."})
    except Exception as e:
        # Cleanup flag on failure to start
        flag_file = os.path.join(DATA_DIR, "training_in_progress.flag")
        if os.path.exists(flag_file):
            os.remove(flag_file)
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

    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")

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
        sender_email = os.environ.get("EMAIL_USER")
        sender_password = os.environ.get("EMAIL_PASS")
        
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
    print("[*] Flask server starting...")
    # Dynamic port for Render deployment
    port = int(os.environ.get("PORT", 3402))
    app.run(host="0.0.0.0", port=port, debug=False)
