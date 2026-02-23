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
for pkg in ["punkt", "stopwords", "wordnet"]:
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

# ================= HELPERS =================
def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, "html.parser").get_text().strip()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def is_url(text):
    text = text.strip().lower()
    pattern = r'^(https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+([/?#].*)?$'
    return bool(re.match(pattern, text))

def extract_text_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator=" ").strip()
    except Exception as e:
        print(f"[!] URL extraction failed: {e}")
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
    "bbc.com","voasomali.com","goobjoog.com","garoweonline.com","somalistream.com",
    "somnn.com","somaliglobe.net","sntv.so","sonna.so","aljazeera.com","reuters.com",
    "apnews.com","hiiraan.com"
]

UNTRUSTED_PATTERNS = [
    "shidan","fayras","dawo mucjiso ah","lacag bilaash ah","guji halkan",
    "win iphone","naxdin","deg deg","nin yaaban","naag yaaban",
    "subxaanallaah","yaabka aduunka","arrin lala yaabo","qarax cusub",
    "war hadda soo dhacay","daawasho naxdin leh"
]

def heuristic_fact_check(text, url=None):
    score = 0
    reasons = []
    if url:
        clean_url = re.sub(r'^https?://(www\.)?', '', url.lower())
        for trusted in TRUSTED_SOURCES:
            if trusted in clean_url:
                score += 60
                reasons.append(f"Isha warka ({trusted}) waa mid si weyn loo kalsoon yahay.")
                break
        else:
            reasons.append("Isha warka (Domain) ma ahan mid ka mid ah ilaha rasmiga ee la yaqaano.")
            if any(ext in clean_url for ext in [".tk",".ga",".ml",".cf",".icu",".xyz"]):
                score -= 30
                reasons.append("Domain-ka loo isticmaalo warkaan inta badan waxaa loo isticmaalaa warar been ah.")
    text_lower = text.lower()
    if any(p in text_lower for p in UNTRUSTED_PATTERNS):
        score -= 25
        reasons.append("Waxaa la helay ereyo kicin ah.")
    else:
        score += 10
    words = text.split()
    if len(words)<30:
        score -= 20
    else:
        score += 15
    confidence = 50 + (abs(score)/2)
    if confidence>98: confidence=98
    rating = "Trusted" if score>=20 else "Unverified"
    return {"rating": rating, "confidence": f"{int(confidence)}%", "reasons": reasons}

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status":"OK","message":"Fake News Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error":"JSON lama helin"}),400
        content = data.get("text") or data.get("data")
        if not content:
            return jsonify({"error":"Qoraal lama soo dirin"}),400
        content=str(content).strip()
        input_type = data.get("type","text")
        input_url = None
        if input_type=="url" or is_url(content):
            if not content.startswith(("http://","https://")):
                content="https://"+content
            input_url=content
            extracted=extract_text_from_url(input_url)
            if not extracted and input_url.startswith("https://"):
                input_url=input_url.replace("https://","http://")
                extracted=extract_text_from_url(input_url)
            if not extracted:
                return jsonify({"error":"Ma suurtagalin in xog laga soo saaro URL-ka."}),400
            content=extracted
        clean_input=preprocess_text(content)
        X=vectorizer.transform([clean_input])

        # ✅ FEATURE MISMATCH CHECK
        if hasattr(model,"n_features_in_"):
            if X.shape[1]!=model.n_features_in_:
                return jsonify({"error":"Feature mismatch: Input lama habeyn karo"}),400
            ext=is_extreme_claim(content)
            vague=is_vague_source(content)
            X_dense=X.toarray()
            X=np.hstack([X_dense, np.array([[ext,vague]])])

        score=model.decision_function(X)[0] if hasattr(model,"decision_function") else 0
        trust_boost=0.0
        if input_url:
            h_result=heuristic_fact_check(content,input_url)
            if any(t in input_url.lower() for t in TRUSTED_SOURCES):
                trust_boost+=5.0
            if h_result["rating"]=="Trusted":
                trust_boost+=2.5
            else:
                trust_boost-=2.0
        final_score=score+trust_boost
        confidence_val=(1/(1+np.exp(-abs(final_score))))*100
        confidence_val=min(98.5,max(70.0,confidence_val))
        result="Trusted" if final_score>0 else "Fake Information"
        return jsonify({"prediction":result,"confidence":f"{round(confidence_val,2)}%","hybrid_score":round(final_score,2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay","detail":str(e)}),500

@app.route("/fact-check", methods=["POST"])
def fact_check():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error":"JSON lama helin"}),400
        content=data.get("text") or data.get("data")
        if not content:
            return jsonify({"error":"Xog lama soo dirin"}),400
        input_url=None
        input_type=data.get("type","text")
        if input_type=="url" or is_url(content):
            if not content.startswith(("http://","https://")):
                content="https://"+content
            input_url=content
            content=extract_text_from_url(input_url)
            if not content:
                input_url=input_url.replace("https://","http://")
                content=extract_text_from_url(input_url)
        if not content or len(str(content).strip())<5:
            return jsonify({"error":"Qoraalka laga helay URL-ka lama heli karo ama waa mid aad u yar"}),400
        return jsonify(heuristic_fact_check(content,input_url))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay","detail":str(e)}),500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data=request.get_json(silent=True)
        if not data:
            return jsonify({"error":"Data lama helin"}),400
        name=data.get("name")
        email=data.get("email")
        message=data.get("message")
        if not all([name,email,message]):
            return jsonify({"error":"Fadlan buuxi dhamaan meelaha banaan"}),400
        with open("contacts.txt","a",encoding="utf-8") as f:
            f.write(f"Name:{name}\nEmail:{email}\nMessage:{message}\n---\n")
        return jsonify({"status":"Success","message":"Fariintaada waa nala soo gaarsiiyey!"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay","detail":str(e)}),500

if __name__=="__main__":
    print("[*] Flask server starting...")
    app.run(host="0.0.0.0", port=3402, debug=False)
