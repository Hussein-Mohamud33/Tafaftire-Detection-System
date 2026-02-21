# ================= REAL NEWS SCRAPER (BREADTH-FIRST & FAST) =================

import re
import requests
import random
import pandas as pd
import nltk
import time
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ---------------- NLTK ----------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# ---------------- CLEAN ----------------
def clean_text(text):
    if not text: return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# ---------------- PREPROCESS ----------------
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# ---------------- TRANSLATION ----------------
from deep_translator import GoogleTranslator

def translate_to_somali(text):
    """Turjum qoraalka una bedel Soomaali"""
    try:
        if not text: return ""
        return GoogleTranslator(source='auto', target='so').translate(text[:2000])
    except Exception as e:
        return text

# ---------------- SUBJECT CLASSIFIER ----------------
def auto_subject(text):
    text = text.lower()
    if any(x in text for x in ['doorasho','madaxweyne','dowlad','baarlamaan','politics','election','government']): return 'Politics'
    if any(x in text for x in ['ciidan','qarax','weerar','amni','security','attack','military','bomb']): return 'Security'
    if any(x in text for x in ['caafimaad','isbitaal','cudur','daawo','health','hospital','disease','medicine']): return 'Health'
    if any(x in text for x in ['lacag','bangiga','dollar','ganacsi','finance','bank','money','trade']): return 'Finance'
    return 'General'

# ---------------- NEWS SITES ----------------
SITES = [
    'https://www.voasomali.com/', 'https://www.voasomali.com/z/2443', 'https://www.voasomali.com/z/2444',
    'https://www.aa.com.tr/so', 'https://www.aa.com.tr/so/turkiga', 'https://www.aa.com.tr/so/calamka',
    'https://www.bbc.com/somali/war', 'https://www.bbc.com/somali/topics/c404v027n0jt',
    'https://www.rfi.fr/so/soomaaliya/', 'https://www.rfi.fr/so/afrika/',
    'https://horseedmedia.net/category/warar/', 'https://horseedmedia.net/category/somaliland/',
    'https://wardheernews.com/category/news/', 'https://wardheernews.com/category/opinion/',
    'https://somaliguardian.com/category/warar/', 'https://somaliguardian.com/category/english-news/',
    'https://hornobserver.com/category/somali/', 'https://hornobserver.com/category/opinion/',
    'https://www.hiiraan.com/news/news_archive.aspx', 'https://www.hiiraan.com/news4/default.aspx',
    'https://www.caasimada.net/category/wararka/', 'https://www.caasimada.net/category/tallo-iyo-tusaale/',
    'https://goobjoog.com/category/somali/', 'https://goobjoog.com/category/business/',
    'https://sonna.so/so/category/warar/', 'https://sonna.so/so/category/arimaha-bulshada/',
    'https://sntv.so/category/warar-maanta/', 'https://sntv.so/category/barnaamijyo/',
    'https://www.radiomuqdisho.so/category/warar/', 'https://www.radiomuqdisho.so/category/maqaallo/',
    'https://www.garoweonline.com/index.php/so/warar', 'https://www.garoweonline.com/index.php/so/faallooyin',
    'https://www.puntlandmirror.com/category/warar/', 
    'https://www.shabellemedia.com/category/warar/',
    'https://www.daljir.com/category/warar/',
    'https://punttilandpost.net/category/warar/',
    'https://www.somalinet.com/category/warar/',
    'https://halgan.net/category/warar/',
    'https://www.halbeeg.com/so/category/wararka/',
    'https://www.warsom.com/category/wararka/',
    'https://www.jowhar.com/category/warar/',
    'https://www.dayniile.com/category/warar/',
    'https://www.somtribune.com/category/wararka/',
    'https://www.somaliland.com/category/warar/',
    'https://www.reuters.com/world',
    'https://www.aljazeera.com/news/',
    'https://www.cnn.com/world',
    'https://www.theguardian.com/world'
]

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def get_soup(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=5)
        if r.status_code == 200:
            return BeautifulSoup(r.text, 'html.parser')
    except:
        pass
    return None

def extract_links(url):
    soup = get_soup(url)
    if not soup: return []
    links = []
    
    # Very aggressive title/link extraction
    for item in soup.find_all('a', href=True):
        title = item.get_text(strip=True)
        link = urljoin(url, item['href'])
        
        # Lower text length threshold to 20 for more content
        if title and len(title) > 20:
             links.append((title, link))
             
    # Shuffling to avoid domain hitting
    random.shuffle(links)
    return links

def scrape_real_news(target=3000):
    seen_articles = set()
    initial_count = 0
    
    # Check existing count and load seen titles safely
    if os.path.exists('Real_news.csv'):
        try:
            # Just read titles/links to avoid parsing issues destroying data
            df = pd.read_csv('Real_news.csv', usecols=['title', 'link'], on_bad_lines='skip')
            seen_articles = set(df['title'].tolist())
            initial_count = len(df)
            print(f"Detected {initial_count} existing records.")
        except Exception as e:
            print(f"Error reading existing file: {e}")

    if initial_count >= target:
        print(f"Target already reached: {initial_count} >= {target}")
        return

    scraped_pages = set()
    queue = list(SITES)
    new_rows = []
    
    print(f'Starting Crawler to add {target - initial_count} records...', flush=True)

    with ThreadPoolExecutor(max_workers=500) as executor:
        while queue and (initial_count + len(new_rows)) < target:
            current_batch = []
            while queue and len(current_batch) < 300:
                url = queue.pop(0)
                if url not in scraped_pages:
                    scraped_pages.add(url)
                    current_batch.append(url)
            
            if not current_batch: break

            futures = {executor.submit(extract_links, url): url for url in current_batch}
            
            for future in as_completed(futures):
                try:
                    links = future.result()
                    if links is None: continue
                    for title, url in links:
                        # Discovery logic
                        is_somali_site = any(kw in url for kw in ['somali', '.so', 'hiiraan', 'muqdisho', 'aa.com', 'horseed', 'wardheer', 'guardian', 'horn', 'voa', 'jowhar', 'goobjoog', 'halbeeg'])
                        
                        if any(domain in url for domain in ['bbc.com', '.so', '.net', '.org', '.com', '.edu']) and url not in scraped_pages and url not in queue:
                             if len(queue) < 15000:
                                if is_somali_site:
                                    queue.insert(0, url) # Prioritize Somali sites
                                else:
                                    queue.append(url)

                        if title and len(title) > 15 and title not in seen_articles:
                            seen_articles.add(title)
                            # is_somali_site already calculated above
                            clean_title = clean_text(title)
                            
                            translated_title = title
                            if not is_somali_site:
                                translated_title = translate_to_somali(title)
                                clean_title = clean_text(translated_title)
                                processed = preprocess_text(clean_title)
                            else:
                                processed = preprocess_text(clean_title)
                            
                            subject = auto_subject(clean_title)
                            
                            row = {
                                'link': url,
                                'title': translated_title,
                                'Text': processed,
                                'Subject': subject,
                                'label': 1
                            }
                            
                            new_rows.append(row)
                            
                            current_total = initial_count + len(new_rows)
                            
                            # Batch write every 50 rows to reduce I/O overhead
                            if len(new_rows) % 50 == 0:
                                batch_to_write = new_rows[-50:]
                                pd.DataFrame(batch_to_write).to_csv('Real_news.csv', mode='a', header=False, index=False, encoding='utf-8-sig')
                                print(f'Progress: {current_total}/{target} (Queue: {len(queue)})', flush=True)
                            
                            if current_total >= target: 
                                # Write remaining rows if target reached mid-batch
                                remaining = len(new_rows) % 50
                                if remaining > 0:
                                    pd.DataFrame(new_rows[-remaining:]).to_csv('Real_news.csv', mode='a', header=False, index=False, encoding='utf-8-sig')
                                break
                    if (initial_count + len(new_rows)) >= target: break
                except Exception:
                    pass
            print(f'Batch done. Total: {initial_count + len(new_rows)}', flush=True)

    # Final write of any remaining rows not in last full batch
    if len(new_rows) % 50 != 0 and (initial_count + len(new_rows)) < target:
         remaining = len(new_rows) % 50
         pd.DataFrame(new_rows[-remaining:]).to_csv('Real_news.csv', mode='a', header=False, index=False, encoding='utf-8-sig')

    print(f'SUCCESS: REAL DATASET UPDATED. Total: {initial_count + len(new_rows)}', flush=True)

if __name__ == '__main__':
    scrape_real_news(3000)
