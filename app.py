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
    "waa","iyo","in","uu","ay","ayuu","ayey","ka","u","ee","oo","ah",
    "sidii","waxaan","waxaad","wuxuu","waxay","iska","ahaa","lagu","loogu",
    "isagoo","iyadoo","ku","soo","isaga","iyada","labada","kala","inta",
    "ilaa","wax","kale","mar","markii","la","si","aad","eeg","ayaa",
    "ayay","kuwa","kuwaas","kuwan","kaas","kan","kuwaa","loo","loona"
]
stop_words.update(somali_stopwords)
lemmatizer = WordNetLemmatizer()

# ================= HELPERS =================
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z' ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w)>2]
    return " ".join(tokens)

def is_url(text):
    text = str(text).strip().lower()
    pattern = r'^(https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+([/?#].*)?$'
    return bool(re.match(pattern, text))

def extract_text_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200: return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for script in soup(["script","style"]): script.decompose()
        text = soup.get_text(separator=" ")
        return text.strip()
    except Exception:
        return ""

# ================= LOAD MODEL =================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR,"saved_model","svm_high_confidence.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR,"saved_model","fake_real_TF_IDF_vectorizer.pkl")
    ENCODER_PATH = os.path.join(BASE_DIR,"saved_model","fake_real_label_encoder.pkl")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    traceback.print_exc()
    exit(1)

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

        content = str(content).strip()
        input_type = data.get("type","text")
        input_url = None

        # ========= URL EXTRACTION =========
        if input_type=="url" or is_url(content):
            if not content.startswith(("http://","https://")):
                content="https://"+content
            input_url=content
            extracted=extract_text_from_url(input_url)
            if not extracted and input_url.startswith("https://"):
                input_url=input_url.replace("https://","http://")
                extracted=extract_text_from_url(input_url)
            if not extracted:
                return jsonify({"error":"Ma suurtagalin in xog laga soo saaro URL-ka"}),400
            content = extracted

        # ========= PREPROCESS =========
        clean_input = preprocess_text(content)
        if not clean_input:
            return jsonify({"error":"Qoraalka kadib preprocessing waa madhan"}),400

        # ========= VECTORIZE =========
        X = vectorizer.transform([clean_input])
        if X.shape[1]!=model.n_features_in_:
            return jsonify({"error":"Feature mismatch between vectorizer and model"}),500
        X = X.toarray()

        # ========= PREDICTION =========
        prediction = model.predict(X)[0]
        confidence = float(max(model.predict_proba(X)[0])*100) if hasattr(model,"predict_proba") else None

        return jsonify({"prediction":str(prediction),"confidence":round(confidence,2) if confidence else None})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay","detail":str(e)}),500

# ================= RUN SERVER =================
if __name__=="__main__":
    port=int(os.environ.get("PORT",3402))
    app.run(host="0.0.0.0",port=port,debug=True)
