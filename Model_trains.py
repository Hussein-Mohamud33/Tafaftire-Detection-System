import pandas as pd
import nltk
import re
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

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

# Data storage synced with app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "system_data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)


# ======================================
# NLTK SETUP
# ======================================
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
# Add Somali stopwords
somali_stopwords = [
    "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
    "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
    "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
    "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
    "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona",
    "yahay", "yihiin", "ahayd", "ahaa", "noqday", "noqon", "leh", "leeyihiin",
    "kala", "hore", "danbe", "dhammaan", "kasta", "badnaa", "yar", "weyn",
    "oo", "kale", "jira", "jiray", "ilaa", "halkan", "halkaas", "mid", "kaliya",
    "isla", "markaana", "ahaana", "ahaanna", "hadda", "horey", "sheegay", "sheegtay"
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
    return None

print("Loading datasets...")

fake_path = find_file("Fake-news.csv")
real_path = find_file("Real-news.csv")

if not fake_path or not real_path:
    print("Dataset lama helin")
    exit(1)

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

# Ensure 'Text' column is string
fake_df["Text"] = fake_df["Text"].astype(str)
real_df["Text"] = real_df["Text"].astype(str)

# ======================================
# DATA DE-DUPLICATION (Improvement)
# ======================================
print("Removing duplicates before training...")
fake_df = fake_df.drop_duplicates(subset=['Text']).fillna("")
real_df = real_df.drop_duplicates(subset=['Text']).fillna("")

texts = pd.concat([fake_df["Text"], real_df["Text"]])
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
# Improvment: Use n-gram range (1, 3) and slightly better min_df to reduce noise
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=3, use_idf=True, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Add extreme/vague features to TF-IDF sparse matrix
from scipy.sparse import hstack
X_train_tfidf = hstack([X_train_tfidf, np.array([ext_train, vague_train]).T])
X_test_tfidf = hstack([X_test_tfidf, np.array([ext_test, vague_test]).T])

# ======================================
# MODELS
# ======================================
# Using class_weight='balanced' to handle the heavy imbalance in the dataset
# Using class_weight='balanced' to handle the heavy imbalance in the dataset

models = {
    "Naive_Bayes": MultinomialNB(alpha=0.01), # Lower alpha for more sensitivity
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
# CREATE SAVE FOLDER
# ======================================
os.makedirs("saved_model", exist_ok=True)

# ======================================
# SAVE ALL MODELS
# ======================================
for name, model in trained_models.items():
    filename = f"saved_model/{name.lower()}_model.pkl"
    joblib.dump(model, filename)
    print(f"Saved: {filename}")

# Save vectorizer & encoder
joblib.dump(tfidf, "saved_model/fake_real_TF_IDF_vectorizer.pkl")
joblib.dump(le, "saved_model/fake_real_label_encoder.pkl")

# ======================================
# BEST MODEL (HIGH CONFIDENCE) 
# ======================================
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

# Fit on full data
X_full_tfidf = tfidf.transform(processed_texts)
X_full_tfidf = hstack([X_full_tfidf, np.array([extreme_flags, vague_flags]).T])
best_model.fit(X_full_tfidf, y)

# explicitly save SVM model to match expected app.py path, irrespective of best_model.
high_model_path = "saved_model/svm_high_confidence.pkl"
svm_model = trained_models.get("SVM", best_model)
svm_model.fit(X_full_tfidf, y)
joblib.dump(svm_model, high_model_path)
joblib.dump(tfidf, "saved_model/fake_real_TF_IDF_vectorizer.pkl")
joblib.dump(le, "saved_model/fake_real_label_encoder.pkl")

import json
stats_file = os.path.join(DATA_DIR, "stats.json")
stats = {}
if os.path.exists(stats_file):
    try:
        with open(stats_file, "r") as f:
            stats = json.load(f)
    except:
        pass
best_acc = results[best_model_name] * 100
stats["model_accuracy"] = f"{best_acc:.1f}%"
with open(stats_file, "w") as f:
    json.dump(stats, f)

print(f"\nBest Model: {best_model_name}")
print(f"High confidence model saved as: {high_model_path}")

# ======================================
# ACCURACY TABLE IMAGE
# ======================================
df_results = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
df_results = df_results.sort_values(by="Accuracy", ascending=False)

fig, ax = plt.subplots(figsize=(7, 3))
ax.axis('off')

table = ax.table(
    cellText=df_results.values,
    colLabels=df_results.columns,
    cellLoc='center',
    loc='center'
)

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

print("Accuracy table saved: saved_model/model_accuracy_table.png")
print("\nDHAMAAN HAWLII WAA LA DHAMEEYSTIRAY")

# Automatic Cleanup of training flag
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--flag", help="Path to the training flag file")
args, unknown = parser.parse_known_args()

if args.flag and os.path.exists(args.flag):
    try:
        os.remove(args.flag)
        print(f"[*] Training flag removed: {args.flag}")
    except Exception as e:
        print(f"[!] Error removing flag: {e}")

