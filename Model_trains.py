import pandas as pd
import nltk
import re
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from scipy.sparse import hstack

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Define DATA_DIR outside the workspace to prevent Live Server reloads
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ======================================
# NLTK SETUP
# ======================================
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("[*] Downloading NLTK resources...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
# Add Somali stopwords (Expanded for better accuracy)
somali_stopwords = [
    "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
    "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
    "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
    "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
    "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona",
    "yahay", "yihiin", "ahayd", "ahaa", "noqday", "noqon", "leh", "leeyihiin",
    "kala", "hore", "danbe", "dhammaan", "kasta", "badnaa", "yar", "weyn",
    "waxa", "waxaa", "ila", "mid", "halkaas", "halkan", "door", "qaatay",
    "kaasoo", "ayadoo", "isagaa", "iyadaa", "kuwaasoo", "hadana", "maxaa", "maxay"
]
stop_words.update(somali_stopwords)

lemmatizer = WordNetLemmatizer()

# ======================================
# TEXT PREPROCESSING
# ======================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep A-Z and apostrophes for Somali/English
    text = re.sub(r"[^a-z' ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ======================================
# EXTREME / VAGUE CLAIM DETECTION
# ======================================
def is_extreme_claim(text):
    if not isinstance(text, str): return 0
    extreme_words = ["100 sano", "hal charge 6 bilood", "miracle", "cure", "mucjiso", "lacag bilaash"]
    return int(any(word in text.lower() for word in extreme_words))

def is_vague_source(text):
    if not isinstance(text, str): return 0
    vague_words = ["khubaro ayaa sheegay", "daraasad cusub ayaa sheegtay", "ilo wareedyo", "warar la helayo"]
    return int(any(word in text.lower() for word in vague_words))

# ======================================
# FIND DATASET
# ======================================
def find_file(filename):
    if os.path.exists(filename):
        return filename
    dataset_path = os.path.join("Dataset", filename)
    if os.path.exists(dataset_path):
        return dataset_path
    # Try case Variations
    if os.path.exists(os.path.join("Dataset", filename.lower())):
        return os.path.join("Dataset", filename.lower())
    return None

def main():
    print("Loading datasets...")

    fake_path = find_file("Fake-news.csv")
    real_path = find_file("Real-news.csv")

    if not fake_path or not real_path:
        # Fallback check
        fake_path = find_file("fake-news.csv")
        if not fake_path:
            print("Dataset lama helin")
            return

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # Ensure 'Text' column is string
    fake_df["Text"] = fake_df["Text"].astype(str)
    real_df["Text"] = real_df["Text"].astype(str)

    # ======================================
    # CLEAN & PREPARE DATA
    # ======================================
    # CRITICAL: Drop duplicates to prevent over-fitting on repetitive data
    fake_df = fake_df.drop_duplicates(subset=["Text"])
    real_df = real_df.drop_duplicates(subset=["Text"])
    
    print(f"Unique Fake: {len(fake_df)} | Unique Real: {len(real_df)}")

    texts = pd.concat([fake_df["Text"], real_df["Text"]])
    # Use the 'Label' column if exists, otherwise fallback to positional
    if "Label" in fake_df.columns and "Label" in real_df.columns:
        labels = pd.concat([fake_df["Label"], real_df["Label"]])
    else:
        labels = [0] * len(fake_df) + [1] * len(real_df)

    print("Preprocessing text...")
    processed_texts = [preprocess_text(t) for t in texts]

    # Add extreme/vague features
    extreme_flags = [is_extreme_claim(t) for t in texts]
    vague_flags = [is_vague_source(t) for t in texts]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # ======================================
    # SPLIT DATA
    # ======================================
    X_train, X_test, y_train, y_test, ext_train, ext_test, vague_train, vague_test = train_test_split(
        processed_texts, y, extreme_flags, vague_flags, test_size=0.2, random_state=42
    )

    # ======================================
    # TF-IDF
    # ======================================
    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 3), min_df=2, use_idf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Add extreme/vague features to TF-IDF sparse matrix
    X_train_tfidf = hstack([X_train_tfidf, np.array([ext_train, vague_train]).T])
    X_test_tfidf = hstack([X_test_tfidf, np.array([ext_test, vague_test]).T])

    # ======================================
    # MODELS
    # ======================================
    models = {
        "Naive_Bayes": MultinomialNB(alpha=0.01),
        "SVM": LinearSVC(max_iter=15000, C=1.0, dual=True, class_weight='balanced'),
        "Logistic_Regression": LogisticRegression(max_iter=4000, solver='lbfgs', class_weight='balanced')
    }

    results = {}
    trained_models = {}

    print("\n===== MODEL RESULTS =====")

    for name, model in models.items():
        print(f"\nTraining {name}")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, preds)
        results[name] = acc
        trained_models[name] = model

        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

    # ======================================
    # SAVE MODELS
    # ======================================
    os.makedirs("saved_model", exist_ok=True)
    for name, model in trained_models.items():
        filename = f"saved_model/{name.lower()}_model.pkl"
        joblib.dump(model, filename)
        print(f"Saved: {filename}")

    # Best Model Logic
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    # Fit on full data
    X_full_tfidf = tfidf.transform(processed_texts)
    X_full_tfidf = hstack([X_full_tfidf, np.array([extreme_flags, vague_flags]).T])
    best_model.fit(X_full_tfidf, y)

    # Save vectorizer & encoder
    joblib.dump(tfidf, "saved_model/fake_real_TF_IDF_vectorizer.pkl")
    joblib.dump(le, "saved_model/fake_real_label_encoder.pkl")

    # Sync with app.py expected path
    high_model_path = "saved_model/svm_high_confidence.pkl"
    svm_model = trained_models.get("SVM", best_model)
    svm_model.fit(X_full_tfidf, y)
    joblib.dump(svm_model, high_model_path)

    # Update stats
    stats_file = os.path.join(DATA_DIR, "stats.json")
    stats = {}
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
        except: pass
        
    best_acc = results[best_model_name] * 100
    stats["model_accuracy"] = f"{best_acc:.1f}%"
    with open(stats_file, "w") as f:
        json.dump(stats, f)

    # ======================================
    # ACCURACY TABLE IMAGE
    # ======================================
    df_results = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    df_results = df_results.sort_values(by="Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    table = ax.table(cellText=df_results.values, colLabels=df_results.columns, cellLoc='center', loc='center')
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4CAF50")
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor("#E3F2FD" if row % 2 == 0 else "#BBDEFB")
    table.scale(1, 1.5)
    plt.title("Model Accuracy Comparison", fontsize=12, fontweight="bold")
    plt.savefig("saved_model/model_accuracy_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nDHAMAAN HAWLII WAA LA DHAMEEYSTIRAY")

if __name__ == "__main__":
    main()
