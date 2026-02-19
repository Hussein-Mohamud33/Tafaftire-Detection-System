# ================= REAL NEWS SCRAPER (BREADTH-FIRST & FAST) =================

import re
import requests
import random
import pandas as pd
import nltk
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    if not text: return ""
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

# ---------------- SUBJECT CLASSIFIER ----------------
def auto_subject(text):
    text = text.lower()
    if any(x in text for x in ["doorasho","madaxweyne","dowlad","baarlamaan"]): return "Politics"
    if any(x in text for x in ["ciidan","qarax","weerar","amni"]): return "Security"
    if any(x in text for x in ["caafimaad","isbitaal","cudur","daawo"]): return "Health"
    if any(x in text for x in ["lacag","bangiga","dollar","ganacsi"]): return "Finance"
    return "General"

# ---------------- NEWS SITES ----------------
SITES = [
    "https://www.bbc.com/somali",
    "https://sonna.so/so",
    "https://sntv.so",
    "https://www.garoweonline.com/index.php/so",
    "https://goobjoog.com/",
    "https://www.caasimada.net/",
    "https://wararka24.com/",
    "https://mudug24.com/",
    "https://radiomuqdisho.so/",
    "https://puntlandpost.net/",
    "https://shabellemedia.com/category/warka/",
    "https://www.hiiraanweyn.net/",
    "https://jubalandtv.com/",
    "https://jowhar.com/",
    "https://universaltvsomali.net/",
    "https://www.voasomali.com/",
    "https://horseedmedia.net/so/",
    "https://mogadishucentre.com/so/",
    "https://dayniile.com/",
    "https://wadani.so/",
    "https://warsimid.com/",
    "https://dhatko.com/",
    "https://allceel.com/",
    "https://halsan.com/",
    "https://dhamays.com/"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def get_soup(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
    except:
        pass
    return None

def extract_links(url):
    soup = get_soup(url)
    if not soup: return []
    links = []
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        link = urljoin(url, a["href"])
        if title and len(title) > 30:
            links.append((title, link))
    return links

def scrape_real_news(target=2000):
    rows = []
    seen_articles = set()
    scraped_pages = set()
    queue = list(SITES)
    
    print(f"Starting Breadth-First Crawler (Target: {target})...")

    with ThreadPoolExecutor(max_workers=30) as executor:
        while queue and len(rows) < target:
            # Batch process the current queue
            current_batch = []
            while queue and len(current_batch) < 100:
                url = queue.pop(0)
                if url not in scraped_pages:
                    scraped_pages.add(url)
                    current_batch.append(url)
            
            if not current_batch: break

            futures = {executor.submit(extract_links, url): url for url in current_batch}
            
            for future in as_completed(futures):
                try:
                    links = future.result()
                    for title, url in links:
                        # Add to queue if it's an internal link and we haven't scraped it
                        if any(domain in url for domain in ["bbc.com", ".so", ".net", ".org", ".com"]) and url not in scraped_pages and url not in queue:
                            if len(queue) < 10000: # Limit queue size
                                queue.append(url)

                        # Process as article
                        if title not in seen_articles:
                            seen_articles.add(title)
                            
                            clean_title = clean_text(title)
                            processed = preprocess_text(clean_title)
                            subject = auto_subject(clean_title)
                            
                            rows.append({
                                "Title": title,
                                "Text": processed,
                                "Subject": subject,
                                "Label": 1
                            })
                            
                            if len(rows) % 100 == 0:
                                print(f"Progress: {len(rows)}/{target} (Queue size: {len(queue)})")
                                # Save partial
                                if len(rows) % 500 == 0:
                                    pd.DataFrame(rows).to_csv("Real_news.csv", index=False, encoding="utf-8-sig")
                            
                            if len(rows) >= target: break
                    if len(rows) >= target: break
                except:
                    pass
            
            # Print status after each batch
            print(f"Completed batch. Current rows: {len(rows)}. Queue: {len(queue)}")

    df = pd.DataFrame(rows)
    df.to_csv("Real_news.csv", index=False, encoding="utf-8-sig")
    print(f"SUCCESS: REAL DATASET CREATED. Total: {len(df)}")
    return df

if __name__ == "__main__":
    scrape_real_news(2000)