# ================= FAKE NEWS GENERATOR =================

import re
import random
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------------- NLTK ----------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# ---------------- CLEAN ----------------
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(text.split())

# ---------------- PREPROCESS ----------------
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ---------------- DATA ----------------
CITIES = [
    "Muqdisho", "Hargeisa", "Garowe", "Boosaaso", "Kismaayo", 
    "Baydhabo", "Beledweyne", "Galkacyo", "Borama", "Berbera",
    "Jowhar", "Merca", "Afgoye", "Dhusamareeb", "Eyl",
    "Las Anod", "Erigavo", "Burao", "Sheikh", "Gabiley",
    "Baardheere", "Luuq", "Balanbale", "Cadaado", "Hobyo",
    "Jilib", "Baraawe", "Qoryooley", "Buurhakaba", "Doolow"
]
ENTITIES = [
    "NISA", "Bangiga Dhexe", "Wasaaradda Arrimaha Gudaha", 
    "Madaxtooyada", "Baarlamaanka", "Wasaaradda Maaliyadda",
    "Guddiga Doorashooyinka", "Maxkamadda Sare", "Ciidanka Xoogga",
    "Wasaaradda Caafimaadka"
]
TRIGGERS = [
    "War DEG DEG ah", "Sir culus", "Dhacdo lama filaan ah",
    "Xog hoose", "War hadda soo dhacay", "Bayaan rasmi ah",
    "Go'aan lama filaan ah", "Kulan xasaasi ah", "Wararka saaka",
    "Falanqeyn muhiim ah"
]

SUBJECTS = ["Politics", "Security", "Health", "Finance"]

TEMPLATES = [
    ("{trigger}: {entity} oo {city} go'aan cusub ku dhawaaqay",
     "Warar laga helay {city} ayaa sheegaya in {entity} uu qaaday tallaabo cusub"),
    ("{trigger}: Isbedel ku yimid {entity} ee magaalada {city}",
     "Xaaladda {city} ayaa isbedeshay kadib markii {entity} uu soo saaray amar"),
    ("{trigger}: {city} oo laga dareemayo dhaqdhaqaaq ka socda {entity}",
     "Dadka deegaanka {city} ayaa soo sheegaya in {entity} uu bilaabay hawlgal"),
    ("{trigger}: Shaqaalaha {entity} oo {city} ku yeeshay shir muhiim ah",
     "Magaalada {city} waxaa maanta ku shiray xubno ka tirsan {entity}")
]

# ---------------- GENERATE ----------------
def generate_fake_news(n=2000):

    rows = []
    seen = set()

    while len(rows) < n:

        tpl = random.choice(TEMPLATES)
        city = random.choice(CITIES)
        entity = random.choice(ENTITIES)
        trigger = random.choice(TRIGGERS)
        subject = random.choice(SUBJECTS)

        title = tpl[0].format(city=city, entity=entity, trigger=trigger)
        body = tpl[1].format(city=city, entity=entity, trigger=trigger)

        text = clean_text(title + " " + body)

        if text in seen:
            continue
        seen.add(text)

        rows.append({
            "Title": title,
            "Text": preprocess_text(text),
            "Subject": subject,
            "Label": 0   # Fake
        })

    df = pd.DataFrame(rows)
    df.to_csv("Fake_news.csv", index=False, encoding="utf-8-sig")

    print("Success: Fake News Saved")
    return df


if __name__ == "__main__":
    generate_fake_news(2000)
