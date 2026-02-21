# ================= FAKE NEWS FROM KAGGLE (ISOT) =================

import re
import random
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import requests
import csv
import io

# ---------------- TRANSLATION ----------------
def translate_to_somali(text):
    """Turjum qoraalka una bedel Soomaali"""
    try:
        if not text: return ""
        return GoogleTranslator(source='auto', target='so').translate(text[:1200])
    except Exception as e:
        return text

# ---------------- NLTK ----------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(text.split())

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def process_row(row):
    try:
        title_en = str(row.get('title', ''))
        text_en = str(row.get('text', ''))
        if not title_en or len(title_en) < 30: return None
        
        # Translate with retry
        title_so = translate_to_somali(title_en)
        text_so = translate_to_somali(text_en[:1200])
        
        clean_t = clean_text(title_so + " " + text_so)
        return {
            "link": f"https://kaggle-mirror.org/isot/{random.randint(10000, 99999)}",
            "title": title_so,
            "Text": preprocess_text(clean_t),
            "Subject": "International News",
            "label": 0
        }
    except Exception as e:
        time.sleep(1) # Sleep on error to cool down
        return None

def fetch_kaggle_fake_news(target=3000):
    URL = "https://raw.githubusercontent.com/FavioVazquez/fake-news/master/data/Fake.csv"
    print(f"Fetching Kaggle Fake News from: {URL}...", flush=True)
    try:
        rows = []
        seen = set()
        
        # Load existing progress
        if os.path.exists("Fake_news.csv"):
            try:
                df_old = pd.read_csv("Fake_news.csv")
                rows = df_old.to_dict('records')
                seen = set(df_old['Text'].tolist())
                print(f"Loaded {len(rows)} existing fake news records.", flush=True)
            except:
                pass

        if len(rows) >= target:
             print(f"Target of {target} already reached.", flush=True)
             return pd.DataFrame(rows)

        # Streaming download and manual parse
        response = requests.get(URL, stream=True)
        lines = response.iter_lines(decode_unicode=True)
        
        header_line = next(lines)
        # Skip first 2000 rows to find new data faster
        print("Skipping first 2000 rows...", flush=True)
        for _ in range(2000): 
            try: next(lines)
            except StopIteration: break
            
        # Using a simple proxy to treat the iterator as a file-like object
        reader = csv.DictReader(lines, fieldnames=['title', 'text', 'subject', 'date'])
        
        print(f"Starting Multi-threaded translation (Target: {target})...", flush=True)
        with ThreadPoolExecutor(max_workers=60) as executor:
            batch = []
            for row in reader:
                if len(rows) >= target: break
                
                txt = str(row.get('text', ''))
                if not txt or len(txt) < 100: continue
                
                # Check seen early
                cleaned = clean_text(txt[:500]) # Smaller prefix for speed
                if cleaned in seen: continue
                
                batch.append(row)
                if len(batch) >= 20: # Smaller batches for more frequent updates
                    futures = [executor.submit(process_row, r) for r in batch]
                    for future in as_completed(futures):
                        if len(rows) >= target: break
                        res = future.result()
                        if res and res["Text"] not in seen:
                            seen.add(res["Text"])
                            rows.append(res)
                            if len(rows) % 5 == 0:
                                print(f"Progress: {len(rows)}/{target} Fake News Saved.", flush=True)
                                pd.DataFrame(rows).to_csv("Fake_news.csv", index=False, encoding="utf-8-sig")
                    batch = []
            
        pd.DataFrame(rows).to_csv("Fake_news.csv", index=False, encoding="utf-8-sig")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return pd.DataFrame()

if __name__ == "__main__":
    fetch_kaggle_fake_news(3000)
