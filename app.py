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
import random
import csv
import smtplib
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
import pandas as pd
from scipy.sparse import issparse

# ================= FLASK INIT =================
app = Flask(__name__, static_folder='Front_End', static_url_path='')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.secret_key = os.environ.get("SECRET_KEY", "tafaftire-fallback-secret-key")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

        print(f"[âœ…] HISTORY SAVED: {data_type} | ID: {item_id} | Input: {clean_input[:30]}...")
        return True
    except Exception as e:
        print(f"[âŒ] ERROR SAVING HISTORY: {traceback.format_exc()}")
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
            "link": f"{link}"[:500],
            "title": f"{title}"[:200],
            "Text": f"{text}",
            "Subject": f"{subject}"[:100],
            "label": numerical_label
        }
        
        df_new = pd.DataFrame([new_data])
        
        # Append to CSV
        if os.path.exists(dataset_path):
            df_new.to_csv(dataset_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(dataset_path, index=False, encoding='utf-8-sig')
            
        print(f"[ðŸ“ˆ] DATASET UPDATED: Added new entry to {dataset_name} | Title: {title[:30]}...")
        
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
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

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
    """High-accuracy preprocessing using NLTK word_tokenize."""
    text = text.lower()
    text = CLEAN_TEXT_PATTERN.sub(" ", text)
    tokens = word_tokenize(text) # AI-du tan ayay ku tababarantay
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(cleaned_tokens)

def is_url(text):
    """Fast URL detection."""
    if not isinstance(text, str): return False
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

def extract_text_from_url(url):
    """Ka soo saar qoraalka bogga webka URL si qoto dheer"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        resp = requests.get(url, headers=headers, timeout=12)
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
        # Return empty values rather than raising to allow the system to handle it gracefully
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

# ================= LOAD MODELS (Lazy Loading) =================
_model = None
_vectorizer = None
_label_encoder = None

def get_model_assets():
    global _model, _vectorizer, _label_encoder
    if _model is None:
        try:
            MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "svm_high_confidence.pkl")
            VECTORIZER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_TF_IDF_vectorizer.pkl")
            ENCODER_PATH = os.path.join(BASE_DIR, "saved_model", "fake_real_label_encoder.pkl")

            _model = joblib.load(MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            _label_encoder = joblib.load(ENCODER_PATH)
            print("[âœ…] AI Models loaded successfully")
        except Exception as e:
            print(f"[âŒ] Model loading failed: {e}")
            _model, _vectorizer, _label_encoder = False, False, False
    return _model, _vectorizer, _label_encoder
# ================= HEURISTIC FACT CHECKER =================
TRUSTED_SOURCES = [
    "bbc.com", "voasomali.com", "goobjoog.com", 
    "garoweonline.com", "somalistream.com", "somnn.com", 
    "somaliglobe.net", "sntv.so", "sonna.so", "aljazeera.com",
    "reuters.com", "apnews.com", "hiiraan.com"
]

UNTRUSTED_PATTERNS = [
    "shidan", "fayras", "dawo mucjiso ah", "lacag bilaash ah", 
    "guji halkan", "win iphone", "naxdin", "deg deg", "nin yaaban",
    "naag yaaban", "subxaanallaah", "yaabka aduunka", "arrin lala yaabo",
    "qarax cusub", "war hadda soo dhacay", "daawasho naxdin leh"
]

def heuristic_fact_check(text, url=None):
    """
    Analyzes news credibility based on source reputation, content patterns, 
    and stylistic markers (sensationalism).
    """
    score = 0
    reasons = []
    
    # 1. Source Reliability (Max +60)
    if url:
        url_lower = url.lower()
        clean_url = re.sub(r'^https?://(www\.)?', '', url_lower)
        
        found_trusted = False
        for trusted in TRUSTED_SOURCES:
            if trusted in clean_url:
                found_trusted = True
                score += 60
                reasons.append(f"The news source ({trusted}) is highly trusted.")
                break

        
        if not found_trusted:
            reasons.append("The news source (Domain) is not among known official sources.")

            # Penalize slightly for suspicious domains (e.g., .tk, .ga, .icu)
            if any(ext in clean_url for ext in [".tk", ".ga", ".ml", ".cf", ".icu", ".xyz"]):
                score -= 30
                reasons.append("The domain used for this news (xyz/tk/ml) is often used for fake news.")


    # 2. Sensationalism & Clickbait (Max -40)
    text_lower = text.lower()
    found_scary = [p for p in UNTRUSTED_PATTERNS if p in text_lower]
    if found_scary:
        score -= 40 # Increased from -25
        reasons.append(f"Found sensational words outside journalistic ethics: {', '.join(found_scary)}.")
    else:
        score += 5
        reasons.append("The text does not appear sensational (Professional tone).")


    # 3. Punctuation Analysis (Sensationalism)
    if "!!!" in text or "???" in text:
        score -= 35 # Increased from -15
        reasons.append("Excessive punctuation used to manipulate reader emotions.")

    
    # 4. Capitalization Check (Shouting)
    # Check if more than 20% of words are ALL CAPS
    words = text.split()
    if len(words) > 5:
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if (len(caps_words) / len(words)) > 0.2:
            score -= 30 # Increased from -15
            reasons.append("The text is written in all caps (Shouting), which indicates a lack of professionalism.")


    # 5. Consensus Keywords (Max +30)
    consensus_keywords = [
        "wadahadal", "shir", "madaxweyne", "rayga", "amniga", "shaqo", 
        "cusub", "gobolka", "isgaarsiinta", "waxbarashada", "caafimaadka",
        "baarlamaanka", "doorashooyinka"
    ]
    found_consensus = [w for w in consensus_keywords if w in text_lower]
    if len(found_consensus) >= 3:
        score += 20
        reasons.append("The news subject appears consistent with official news content.")
    elif len(found_consensus) == 0:
        score -= 10
        reasons.append("No significant keywords linking this news to major events found.")


    # 6. Text Length & Quality
    if len(words) < 30:
        score -= 20
        reasons.append("The text is very short and appears not to be fully researched.")
    else:

        score += 15

    # Determine Rating & Confidence
    confidence = 50 + (abs(score) / 2)
    if confidence > 98: confidence = 98

    if score >= 15:
        rating = "Trusted"
    elif score <= -10: # Lowered threshold to catch suspicious better
        rating = "Suspicious" 
        if confidence < 75: confidence = 80
    else:
        rating = "Unverified"
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
    global global_stats
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
            snippet_text = str(content)
            if len(snippet_text) > 60:
                page_title = f"{snippet_text[:60]}..."
            else:
                page_title = snippet_text
        
        # Guess subject
        news_subject = guess_subject(content)

        # ================= Preprocess =================
        clean_input = preprocess_text(content)

        # Lazy load models
        model, vectorizer, label_encoder = get_model_assets()
        if not model:
            return jsonify({"error": "AI model is currently unavailable"}), 503

        # Vectorize
        # Ensure clean_input is a list for transform
        X = vectorizer.transform([str(clean_input)])
        ext = is_extreme_claim(content)
        vague = is_vague_source(content)
        
        # Convert to array for stacking if it's sparse
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        X = np.hstack([X_dense, np.array([[ext, vague]])])

        # ================= Hybrid Decision Logic =================
        # 1. Base AI Score (LinearSVC decision function returns distance from hyperplane)
        score = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0
        
        # 2. Heuristic Check (Expert System Integration)
        trust_boost = 0.0
        
        # Helitaanka dhibcaha heuristic si loo xoojiyo AI-da
        h_result = heuristic_fact_check(content, input_url)
        
        # Boost for quality content (Professional tone)
        if h_result["rating"] == "Trusted":
            trust_boost += 1.5
        
        if input_url:
            # Check if source is explicitly trusted (massive boost)
            is_verified_domain = any(t in input_url.lower() for t in TRUSTED_SOURCES)
            if is_verified_domain:
                trust_boost += 5.0
            
            if h_result["rating"] != "Trusted":
                # If heuristic finds bad patterns, penalize
                trust_boost -= 2.5

        # Final Combined Score (Hybrid Verdict)
        # Haddii dhibcuhu ka badan yihiin -0.5, waa Trusted (si loo yareeyo qaladaadka)
        final_score = score + trust_boost
        
        # Sigmoid function to normalize confidence between 0-100%
        confidence_val = (1 / (1 + np.exp(-abs(final_score)))) * 100
        
        # Cap confidence for reliability
        confidence_val = min(98.5, max(70.0, confidence_val))
        
        # adjustment: -0.5 threshold allows more real news to pass
        is_trusted = final_score > -0.5
        result = "Trusted" if is_trusted else "Fake Information"

        return jsonify({
            "prediction": result, 
            "confidence": f"{round(float(confidence_val), 2)}%",
            "hybrid_score": float(round(float(final_score), 2)), # For internal calibration
            "raw_text": content,
            "title": page_title,
            "link": input_url if input_url else "N/A",
            "subject": news_subject
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error during prediction:", error_msg)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/fact-check", methods=["POST"])
def fact_check():
    global global_stats
    try:
        global_stats["requests_handled"] = global_stats.get("requests_handled", 0) + 1
        save_stats(global_stats)
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON not found"}), 400


        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error": "No data provided"}), 400
        
        # Ensure content is string and clean
        content = str(content).strip()


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
            # Use a snippet of text as title for text inputs
            snippet_text = content
            if len(snippet_text) > 60:
                page_title = f"{snippet_text[:60]}..."
            else:
                page_title = snippet_text

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
        if "trusted" in rating_str: somali_label = "War Rasmi ah"
        elif "suspicious" in rating_str: somali_label = "Shaki Baa Ku Jira"

        return jsonify(fact_result)

    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error during fact-check:", error_msg)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

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
    "username": os.getenv("ADMIN_USER", ""),
    "password": os.getenv("ADMIN_PASS", "")
}
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

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
        return jsonify({"success": True, "token": ADMIN_TOKEN}) # Secure token from env
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
        
        # Prevent manual duplicates
        if os.path.exists(path):
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

    sender_email = os.environ.get("SMTP_EMAIL", "")
    sender_password = os.environ.get("SMTP_PASSWORD", "")

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
        sender_email = os.environ.get("SMTP_EMAIL", "")
        sender_password = os.environ.get("SMTP_PASSWORD", "")
        
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
    print(f"[*] Flask server starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
