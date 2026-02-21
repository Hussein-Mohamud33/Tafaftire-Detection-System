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
app = Flask(__name__, 
            static_folder="Front_End_Data", 
            static_url_path="")
CORS(app)

# ================= NLTK SETUP (Lazy Loading) =================
# We don't download at top-level to avoid Render startup timeouts.
# Packages will be loaded inside the functions that need them.
stop_words = set()
lemmatizer = None

def init_nltk_resources():
    global stop_words, lemmatizer
    if not stop_words:
        packages = ["punkt", "punkt_tab", "stopwords", "wordnet"]
        for pkg in packages:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words("english"))
            somali_stopwords = [
                "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
                "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
                "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
                "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
                "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona"
            ]
            stop_words.update(somali_stopwords)
        except Exception:
            stop_words = set()
            
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
        except Exception:
            lemmatizer = None

# ================= HELPERS =================
def sanitize_text(text):
    """Remove HTML tags and strip text."""
    if not isinstance(text, str):
        return ""
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        pass
    return text.strip()

def preprocess_text(text):
    """Lowercase, remove non-alphabetic, tokenize (with fallback), remove stopwords, lemmatize."""
    init_nltk_resources() # Ensure resources are ready when needed
    
    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    
    # Try NLTK tokenizer, if fails, use split()
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
        
    # Remove stopwords and lemmatize
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    else:
        tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
        
    return " ".join(tokens)

def is_url(text):
    """Detect if input is URL (handles http, https, or www)."""
    text = text.strip().lower()
    pattern = r'^(https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+([/?#].*)?$'
    return bool(re.match(pattern, text))

def extract_text_from_url(url):
    """Ka soo saar qoraalka bogga webka URL"""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
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

# ================= LOAD RESOURCES (Optimized Background) =================
model = None
vectorizer = None
label_encoder = None
models_loaded = False
loading_error = None

def load_resources_in_background():
    global model, vectorizer, label_encoder, models_loaded, stop_words, lemmatizer, loading_error
    try:
        print("[*] Starting resource loading in background...")
        # 1. NLTK Quick Setup
        for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
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

        # 2. Models Loading (Multi-path search)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(BASE_DIR, "saved_model"),
            os.path.join(BASE_DIR, "Saved_model"),
            os.path.join(os.getcwd(), "saved_model"),
            os.path.join(os.getcwd(), "Saved_model")
        ]
        
        models_dir = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                models_dir = path
                break

        if models_dir:
            file_paths = {
                "model": os.path.join(models_dir, "svm_high_confidence.pkl"),
                "vectorizer": os.path.join(models_dir, "fake_real_TF_IDF_vectorizer.pkl"),
                "encoder": os.path.join(models_dir, "fake_real_label_encoder.pkl")
            }
            
            # Verify each file exists
            missing_files = [f for f, p in file_paths.items() if not os.path.exists(p)]
            if not missing_files:
                model = joblib.load(file_paths["model"])
                vectorizer = joblib.load(file_paths["vectorizer"])
                label_encoder = joblib.load(file_paths["encoder"])
                models_loaded = True
                print(f"✅ Background loading complete from: {models_dir}")
            else:
                loading_error = f"Files maqan ({', '.join(missing_files)}) gudaha {models_dir}."
                print(f"❌ ERROR: {loading_error}")
        else:
            loading_error = f"Folder-ka 'saved_model' lama helin. Path-yada la baaray: {possible_paths}"
            print(f"❌ ERROR: {loading_error}")
            
    except Exception as e:
        loading_error = f"Crash dhacay: {str(e)}"
        print(f"❌ Background loading failed: {e}")
        traceback.print_exc()

# Start background loading
import threading
import time
threading.Thread(target=load_resources_in_background, daemon=True).start()
# ================= DEEP FACT CHECKER (Enhanced) =================
TRUSTED_SOURCES = [
    "bbc.com", "voasomali.com", "goobjoog.com", "garoweonline.com", 
    "somalistream.com", "somnn.com", "somaliglobe.net", "sntv.so", 
    "sonna.so", "aljazeera.com", "reuters.com", "apnews.com", 
    "hiiraan.com", "puntlandpost.net", "radiomuqdisho.net", "dalsan.so",
    "caasimada.net", "horseedmedia.net", "somalilandpost.net"
]

UNTRUSTED_PATTERNS = [
    "shidan", "fayras", "dawo mucjiso ah", "lacag bilaash ah", 
    "guji halkan", "win iphone", "naxdin", "deg deg", "nin yaaban",
    "naag yaaban", "subxaanallaah", "yaabka aduunka", "arrin lala yaabo",
    "qarax cusub", "war hadda soo dhacay", "daawasho naxdin leh",
    "lama rumeysan karo", "aad u naxdin badan", "muuqaal sir ah"
]

SOCIAL_MEDIA_DOMAINS = ["facebook.com", "t.me", "x.com", "twitter.com", "tiktok.com", "fb.watch"]

EMOTIONAL_TRIGGERS = [
    "subxaanallaah", "naxdin", "mucjiso", "yaab", "aad u xanuun badan",
    "dunidii way dhammaday", "qarax", "geeri", "deg deg", "nin yaaban"
]

def heuristic_fact_check(text, url=None):
    """
    Advanced Credibility Analysis:
    - URL/Source Authority
    - Linguistic Tone & Emotional Triggers
    - Somali News Standards Consensus
    - Clickbait & Sensationalism Detection
    """
    score = 0
    reasons = []
    text_lower = text.lower()
    
    # 1. Source & Website Deep Check
    if url:
        url_lower = str(url).lower()
        clean_url = re.sub(r'^https?://(www\.)?', '', url_lower)
        
        # A. Trusted Check
        found_trusted = False
        for trusted in TRUSTED_SOURCES:
            if trusted in clean_url:
                found_trusted = True
                score += 70 
                reasons.append(f"✅ Isha rasmiga ah: {trusted}. Tani waa ilo-wareed si weyn looga tixgeliyo saxaafadda Soomaalida.")
                break
        
        # B. Social Media Check (Lower baseline trust)
        if not found_trusted:
            if any(sm in clean_url for sm in SOCIAL_MEDIA_DOMAINS):
                score -= 15
                reasons.append("⚠️ Isha warka waa Social Media. Wararka noocan ah waxay u baahan yihiin in laga xaqiijiyo ilo madax-bannaan.")
            
            # C. Suspicious TLDs
            if any(ext in clean_url for ext in [".tk", ".ml", ".cf", ".ga", ".icu", ".xyz", ".top", ".buzz"]):
                score -= 40
                reasons.append("❌ Domain-ka (xyz/tk/ml): Ciwaanka website-kan waxaa badanaa loo isticmaalaa warar been ah ama 'phishing'.")
            elif not found_trusted:
                reasons.append("ℹ️ Isha warka (Domain): Ciwaankan ma ahan mid ku jira diiwaanka ilaha rasmiga ah ee la yaqaan.")

    # 2. Emotional Tone & Extremism
    found_triggers = [w for w in EMOTIONAL_TRIGGERS if w in text_lower]
    if len(found_triggers) >= 2:
        score -= 25
        reasons.append(f"🚩 Luuqad Kicin ah: Waxaa la isticmaalay ereyo dareenka kiciya sida ({', '.join(found_triggers)}).")
    elif len(found_triggers) == 0:
        score += 15
        reasons.append("✅ Tone Professional: Qoraalku ma lahan ereyo kicin ah, wuxuuna u qoran yahay si dhex-dhexaad ah.")

    # 3. Somali News Consensus (Key Professional Terms)
    professional_terms = [
        "madaxweyne", "baarlamaanka", "doorasho", "xukuumad", "shacabka",
        "amniga", "dowladda", "wada-hadal", "shir-jaraa'id", "go'aan",
        "wasaaradda", "maamulka", "gobolka", "isgaarsiinta", "horumar", "sharciga"
    ]
    found_terms = [w for w in professional_terms if w in text_lower]
    if len(found_terms) >= 4:
        score += 30
        reasons.append("✅ Consensus News: Mowduucu wuxuu adeegsaday Luuqad saxafadeed oo waafaqsan wararka rasmiga ah.")
    elif len(found_terms) >= 1:
        score += 10
        reasons.append("ℹ️ Waxaa ku jira ereyo la xiriira dhacdooyinka rasmiga ah.")

    # 4. Clickbait Markers (Sensationalism)
    if any(p in text_lower for p in UNTRUSTED_PATTERNS):
        score -= 30
        reasons.append("🚩 Clickbait: Waxaa jira calaamado muujinaya in akhristaha loo soo jiidayo si haboonayn.")
    
    # 5. Punctuation (Excessive)
    if "!!!" in text or "???" in text:
        score -= 15
        reasons.append("🚩 Excessive Punctuation: Calaamadaha yaabka iyo su'aalaha badan waxay ka mid yihiin sifooyinka wararka been ah.")

    # 6. Text Depth Analysis
    words = text.split()
    if len(words) > 50:
        score += 20
        reasons.append("✅ Qotodheer: Qoraalku waa mid faahfaahsan, taas oo kordhisa fursadda inuu jiro baaris.")
    elif len(words) < 15:
        score -= 20
        reasons.append("🚩 Qoraal kooban: Qoraalku aad buu u gaaban yahay, lama dhihi karo waa war dhammeystiran.")

    # Final Decision
    confidence = 55 + (abs(score) / 2)
    if confidence > 98: confidence = 98

    if score >= 15:
        rating = "Trusted"
    elif score > -15:
        rating = "Unverified"
        confidence = max(50, confidence - 10)
    else:
        rating = "Unverified"
        if confidence < 70: confidence = 75

    return {
        "rating": rating,
        "confidence": f"{int(confidence)}%",
        "reasons": reasons
    }

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    # Serve Index.html from Front_End_Data
    return app.send_static_file("Index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK" if models_loaded else "ERROR",
        "models_loaded": models_loaded,
        "message": "API is online" if models_loaded else "API is online but models failed to load"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # If not loaded, wait up to 10 seconds (for cold startup)
        wait_time = 0
        while not models_loaded and wait_time < 10:
            if loading_error:
                return jsonify({"error": f"NIDAMKA: Error ayaa dhacay: {loading_error}"}), 500
            time.sleep(1)
            wait_time += 1

        if not models_loaded:
            return jsonify({"error": "NIDAMKA: Model-adii wali ma diyaarsana. Fadlan sug 10 ilbiriqsi ka dibna isku day mar kale."}), 503

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

        # ================= AI Model Prediction Only =================
        # Distance from hyperplane (confidence)
        score = model.decision_function(X)[0] if hasattr(model, "decision_function") else 0
        final_score = score
        
        # Sigmoid function to normalize confidence between 0-100%
        confidence_val = (1 / (1 + np.exp(-abs(final_score)))) * 100
        
        # Cap confidence for reliability
        confidence_val = min(98.5, max(70.0, confidence_val))
        
        is_trusted = final_score > 0
        result = "Trusted" if is_trusted else "Fake Information"
        
        return jsonify({
            "prediction": result, 
            "confidence": f"{round(confidence_val, 2)}%",
            "hybrid_score": round(final_score, 2)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error (Predict): {str(e)}"}), 500

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
        traceback.print_exc()
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
    port = int(os.environ.get("PORT", 3402))
    print(f"[*] Flask server starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
