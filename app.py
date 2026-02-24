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
from bs4 import BeautifulSoup

# ================= FLASK INIT =================
app = Flask(__name__)
CORS(app)

# ================= NLTK SETUP =================
# We try to load resources, and only download if they are missing.
# Downloads are slow on Render, so we use a silent download.
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except (LookupError, AttributeError):
        nltk.download(pkg, quiet=True)


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
    return bool(URL_PATTERN.match(text.strip()))

# Setup a requests session for speed (connection pooling)
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})

def extract_text_from_url(url):
    """Ka soo saar qoraalka bogga webka URL si dhakhso leh"""
    try:
        # Reduced timeout to 4s and use session for pooling
        resp = session.get(url, timeout=4)
        if resp.status_code != 200:
            return ""
        
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Remove unwanted tags
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Get text with a separator
        text = soup.get_text(separator=" ")
        
        return text.strip()
    except Exception:
        return ""


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
                reasons.append(f"Isha warka ({trusted}) waa mid si weyn loo kalsoon yahay.")
                break
        
        if not found_trusted:
            reasons.append("Isha warka (Domain) ma ahan mid ka mid ah ilaha rasmiga ee la yaqaano.")
            # Penalize slightly for suspicious domains (e.g., .tk, .ga, .icu)
            if any(ext in clean_url for ext in [".tk", ".ga", ".ml", ".cf", ".icu", ".xyz"]):
                score -= 30
                reasons.append("Domain-ka loo isticmaalo warkaan (xyz/tk/ml) inta badan waxaa loo isticmaalaa warar been ah.")

    # 2. Sensationalism & Clickbait (Cumulative Penalty)
    text_lower = text.lower()
    found_scary = [p for p in UNTRUSTED_PATTERNS if p in text_lower]
    if found_scary:
        # Subtract 25 for EACH pattern found (Cumulative)
        score -= (len(found_scary) * 25)
        reasons.append(f"Waxaa la helay ereyo kicin ah: {', '.join(found_scary)}.")
    else:
        score += 5

    # 3. Punctuation Analysis (Sensationalism)
    # Catching !!, !!!, ??, ??? or combinations
    if re.search(r'!!|\?\?', text):
        score -= 40
        reasons.append("Waxa la isticmaalay calaamado kicin ah (Excessive punctuation).")
    
    # 4. Capitalization Check (Shouting)
    words = text.split()
    if len(words) > 5:
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if (len(caps_words) / len(words)) > 0.2:
            score -= 30
            reasons.append("Qoraalku wuxuu u qoran yahay si qaylo ah (Too many CAPS).")


    # 5. Consensus Keywords (Max +30)
    consensus_keywords = [
        "wadahadal", "shir", "madaxweyne", "rayga", "amniga", "shaqo", 
        "cusub", "gobolka", "isgaarsiinta", "waxbarashada", "caafimaadka",
        "baarlamaanka", "doorashooyinka"
    ]
    found_consensus = [w for w in consensus_keywords if w in text_lower]
    if len(found_consensus) >= 3:
        score += 20
        reasons.append("Mowduuca warku wuxuu u muuqdaa mid waafaqsan nuxurka wararka rasmiga ah.")
    elif len(found_consensus) == 0:
        score -= 10
        reasons.append("Ma jiraan ereyo muhiim ah oo xiriiriya warkan iyo dhacdooyinka muhiimka ah.")

    # 6. Text Length & Quality
    if len(words) < 30:
        score -= 20
        reasons.append("Qoraalku aad buu u gaaban yahay, wuxuuna u muuqdaa mid aan si buuxda loo baarin.")
    else:
        score += 15

    # Determine Rating & Confidence
    confidence = 50 + (abs(score) / 2)
    if confidence > 98: confidence = 98

    if score >= 20:
        rating = "Trusted"
    elif score <= -15: 
        rating = "Suspicious" 
        confidence = min(98, 75 + abs(score)/2)
    else:
        rating = "Unverified"
        confidence = max(60, 70 - abs(score))

    return {
        "rating": rating,
        "confidence": f"{int(confidence)}%",
        "reasons": reasons,
        "score": score
    }

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "OK", "message": "Fake News Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON lama helin"}), 400

        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error": "Qoraal lama soo dirin"}), 400

        content = str(content).strip()

        input_type = data.get("type", "text")
        input_url = None
        
        # Haddii input uu URL yahay ama ciddida u eg tahay URL
        if input_type == "url" or is_url(content):
            if not content.startswith(("http://", "https://")):
                content = "https://" + content
            
            url_to_extract = content
            input_url = content
            extracted = extract_text_from_url(input_url)
            
            if not extracted and input_url.startswith("https://"):
                input_url = input_url.replace("https://", "http://")
                extracted = extract_text_from_url(input_url)
            
            if not extracted:
                return jsonify({"error": "NIDAMKA: Ma suurtagalin in xog laga soo saaro URL-ka. Fadlan hubi link-ga."}), 400
            content = extracted

        # ================= Preprocess =================
        clean_input = preprocess_text(content)

        # Vectorize
        X = vectorizer.transform([clean_input])
        ext = is_extreme_claim(content)
        vague = is_vague_source(content)
        
        X_dense = X.toarray()
        X = np.hstack([X_dense, np.array([[ext, vague]])])

        # ================= Hybrid Decision Logic =================
        trust_boost = 0.0
        
        # 1. EARLY TRUST CHECK (Fast Path)
        is_verified_domain = False
        if input_url:
            is_verified_domain = any(t in input_url.lower() for t in TRUSTED_SOURCES)
            if is_verified_domain:
                trust_boost += 6.5 # Direct massive boost for verified sources

        # 2. Heuristic Check (Only if not already verified or to check for clickbait)
        # We only run full heuristic if it's NOT a verified domain to save speed,
        # OR we just do a quick check.
        if not is_verified_domain:
            h_result = heuristic_fact_check(content, input_url)
            if h_result["rating"] == "Trusted":
                trust_boost += 1.5
            elif h_result["rating"] == "Suspicious":
                trust_boost -= 2.5
        
        # 3. Base AI Score
        score = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0

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
            "hybrid_score": float(round(float(final_score), 2)) # For internal calibration
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        print("Error during prediction:", error_msg)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/fact-check", methods=["POST"])
def fact_check():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON lama helin"}), 400

        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error": "Xog lama soo dirin"}), 400

        input_url = None
        input_type = data.get("type", "text")

        if input_type == "url" or is_url(content):
            temp_content = content.strip()
            if not temp_content.startswith(("http://", "https://")):
                temp_content = "https://" + temp_content
            
            input_url = temp_content
            content = extract_text_from_url(input_url)
            
            if not content:
                input_url = input_url.replace("https://", "http://")
                content = extract_text_from_url(input_url)

        if not content or len(str(content).strip()) < 5:
            return jsonify({"error": "Qoraalka laga helay URL-ka lama heli karo ama waa mid aad u yar"}), 400

        fact_result = heuristic_fact_check(content, input_url)
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
            return jsonify({"error": "Fadlan buuxi dhamaan meelaha banaan"}), 400

        # Log to file
        with open("contacts.txt", "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")

        print(f"[*] New message from {name} ({email})")
        return jsonify({"status": "Success", "message": "Fariintaada waa nala soo gaarsiiyey!"})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error ayaa dhacay"}), 500

# ================= RUN SERVER =================
if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 3402))
    print(f"[*] Flask server starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

