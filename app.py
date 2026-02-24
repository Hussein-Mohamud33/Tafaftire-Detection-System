import os
import re
import joblib
import traceback
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

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
def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.strip()

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
        for script in soup(["script","style"]):
            script.decompose()
        text = soup.get_text(separator=" ")
        return text.strip()
    except Exception:
        return ""

# ================= HEURISTIC FACT CHECK =================
TRUSTED_SOURCES = [
    "bbc.com","voasomali.com","goobjoog.com",
    "garoweonline.com","somalistream.com","somnn.com",
    "somaliglobe.net","sntv.so","sonna.so","aljazeera.com",
    "reuters.com","apnews.com","hiiraan.com"
]

UNTRUSTED_PATTERNS = [
    "shidan","fayras","dawo mucjiso ah","lacag bilaash ah",
    "guji halkan","win iphone","naxdin","deg deg","nin yaaban",
    "naag yaaban","subxaanallaah","yaabka aduunka","arrin lala yaabo",
    "qarax cusub","war hadda soo dhacay","daawasho naxdin leh"
]

def heuristic_fact_check(text, url=None):
    score = 0
    reasons = []

    # Source reliability
    if url:
        clean_url = re.sub(r'^https?://(www\.)?', '', url.lower())
        found = False
        for trusted in TRUSTED_SOURCES:
            if trusted in clean_url:
                score += 60
                reasons.append(f"Isha warka ({trusted}) waa mid la isku halayn karo")
                found = True
                break
        if not found:
            reasons.append("Isha warka (Domain) lama hubo")
            if any(ext in clean_url for ext in [".tk",".ga",".ml",".cf",".icu",".xyz"]):
                score -= 30
                reasons.append("Domain-ka loo isticmaalo wararka been abuurka ah")

    # Sensationalism
    text_lower = text.lower()
    found_scary = [p for p in UNTRUSTED_PATTERNS if p in text_lower]
    if found_scary:
        score -= 25
        reasons.append(f"Ereyada kicin ah: {', '.join(found_scary)}")
    else:
        score += 10
        reasons.append("Qoraalku wuxuu u muuqdaa mid xirfad leh")

    # Punctuation
    if "!!!" in text or "???" in text:
        score -= 15
        reasons.append("Calaamado badan oo dareen kicinaya")

    # Capital letters
    words = text.split()
    if len(words) > 10:
        caps_words = [w for w in words if w.isupper() and len(w) > 3]
        if (len(caps_words)/len(words)) > 0.3:
            score -= 15
            reasons.append("Qoraal qaylo badan leh (All Caps)")

    # Consensus keywords
    consensus_keywords = ["wadahadal","shir","madaxweyne","rayga","amniga","shaqo",
                          "cusub","gobolka","isgaarsiinta","waxbarashada","caafimaadka",
                          "baarlamaanka","doorashooyinka"]
    found_consensus = [w for w in consensus_keywords if w in text_lower]
    if len(found_consensus) >=3:
        score +=20
        reasons.append("Warku wuxuu u muuqdaa mid rasmi ah")
    elif len(found_consensus) ==0:
        score -=10
        reasons.append("Ma jiraan ereyo muhiim ah oo xiriiriya dhacdooyinka")

    if len(words) <30:
        score -=20
        reasons.append("Qoraal gaaban, aan la hubin")

    confidence = 50 + (abs(score)/2)
    confidence = min(confidence, 98)
    if score >=20: rating="Trusted"
    elif score>-10: rating="Unverified"
    else: rating="Unverified"
    if confidence<70: confidence=75

    return {"rating":rating,"confidence":f"{int(confidence)}%","reasons":reasons}

# ================= LOAD MODELS =================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR,"saved_model","svm_high_confidence.pkl")
    VECT_PATH = os.path.join(BASE_DIR,"saved_model","fake_real_TF_IDF_vectorizer.pkl")
    ENC_PATH = os.path.join(BASE_DIR,"saved_model","fake_real_label_encoder.pkl")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    label_encoder = joblib.load(ENC_PATH)
    print("Models loaded successfully")
except Exception:
    traceback.print_exc()
    model, vectorizer, label_encoder = None, None, None

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status":"OK","message":"Fake News Detection API running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data=request.get_json(silent=True)
        if not data: return jsonify({"error":"JSON lama helin"}),400
        content=data.get("text") or data.get("data")
        if not content: return jsonify({"error":"Qoraal lama soo dirin"}),400
        content = str(content).strip()
        input_type=data.get("type","text")
        input_url=None

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
            content=extracted

        clean_input=preprocess_text(content)
        X=vectorizer.transform([clean_input])
        X=X.toarray()
        score=model.decision_function(X)[0] if hasattr(model,"decision_function") else 0
        trust_boost=0.0
        if input_url:
            h_result=heuristic_fact_check(content,input_url)
            if any(t in input_url.lower() for t in TRUSTED_SOURCES): trust_boost+=5.0
            if h_result["rating"]=="Trusted": trust_boost+=2.5
            else: trust_boost-=2.0
        final_score=score+trust_boost
        confidence_val=(1/(1+np.exp(-abs(final_score))))*100
        confidence_val=min(98.5,max(70,confidence_val))
        result="Trusted" if final_score>0 else "Fake Information"
        return jsonify({"prediction":result,"confidence":round(confidence_val,2),"hybrid_score":round(final_score,2)})
    except Exception:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay"}),500

@app.route("/fact-check", methods=["POST"])
def fact_check_route():
    try:
        data=request.get_json(silent=True)
        if not data: return jsonify({"error":"JSON lama helin"}),400
        content=data.get("text") or data.get("data")
        if not content: return jsonify({"error":"Xog lama soo dirin"}),400
        input_type=data.get("type","text")
        input_url=None
        if input_type=="url" or is_url(content):
            temp_content=content.strip()
            if not temp_content.startswith(("http://","https://")):
                temp_content="https://"+temp_content
            input_url=temp_content
            content=extract_text_from_url(input_url)
            if not content:
                input_url=input_url.replace("https://","http://")
                content=extract_text_from_url(input_url)
        if not content or len(str(content).strip())<5:
            return jsonify({"error":"Qoraalka lama heli karo"}),400
        fact_result=heuristic_fact_check(content,input_url)
        return jsonify(fact_result)
    except Exception:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay"}),500

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        data_dir = "Dataset"
        real_path=os.path.join(data_dir,"Real_news.csv")
        fake_path=os.path.join(data_dir,"Fake_news.csv")
        if not os.path.exists(real_path) or not os.path.exists(fake_path):
            return jsonify({"error":"Dataset files ma jiraan"}),400
        df_real=pd.read_csv(real_path)
        df_fake=pd.read_csv(fake_path)
        df_real["label"]="Real"
        df_fake["label"]="Fake"
        df=pd.concat([df_real,df_fake],ignore_index=True)
        df["text"]=df["text"].astype(str)
        X_texts=[preprocess_text(t) for t in df["text"].tolist()]
        y=df["label"].tolist()
        le=LabelEncoder()
        y_enc=le.fit_transform(y)
        vect=TfidfVectorizer(max_features=5000)
        X=vect.fit_transform(X_texts)
        clf=LinearSVC()
        clf.fit(X,y_enc)
        # Save models
        saved_dir="saved_model"
        os.makedirs(saved_dir,exist_ok=True)
        joblib.dump(clf,os.path.join(saved_dir,"svm_high_confidence.pkl"))
        joblib.dump(vect,os.path.join(saved_dir,"fake_real_TF_IDF_vectorizer.pkl"))
        joblib.dump(le,os.path.join(saved_dir,"fake_real_label_encoder.pkl"))
        return jsonify({"status":"Success","message":"Model dib loo tababaray"})
    except Exception:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay"}),500

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data=request.get_json(silent=True)
        if not data: return jsonify({"error":"Data lama helin"}),400
        name=data.get("name")
        email=data.get("email")
        message=data.get("message")
        if not all([name,email,message]): return jsonify({"error":"Fadlan buuxi meelaha"}),400
        with open("contacts.txt","a",encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")
        return jsonify({"status":"Success"})
    except Exception:
        traceback.print_exc()
        return jsonify({"error":"Server error ayaa dhacay"}),500

# ================= RUN =================
if __name__=="__main__":
    port=int(os.environ.get("PORT",3402))
    app.run(host="0.0.0.0",port=port)
