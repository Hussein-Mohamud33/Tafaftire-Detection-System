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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import json

# Define DATA_DIR outside the workspace to prevent Live Server reloads
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, ".tafaftire_system_data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ================= NLTK SETUP =================
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            path = f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}"
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)
    
    # Explicitly trigger WordNet load to avoid LazyCorpusLoader AttributeError
    from nltk.corpus import wordnet
    try:
        wordnet.ensure_loaded()
    except Exception:
        # Fallback for older NLTK versions
        _ = wordnet.fileids()

setup_nltk()

stop_words = set(stopwords.words("english"))
# Add Somali stopwords
somali_stopwords = [
    "waa", "iyo", "in", "uu", "ay", "ayuu", "ayey", "ka", "u", "ee", "oo", "ah", 
    "sidii", "waxaan", "waxaad", "wuxuu", "waxay", "iska", "ahaa", "lagu", "loogu",
    "isagoo", "iyadoo", "ku", "soo", "isaga", "iyada", "labada", "kala", "inta",
    "ilaa", "wax", "kale", "mar", "markii", "la", "si", "aad", "eeg", "ayaa",
    "ayay", "kuwa", "kuwaas", "kuwan", "kaas", "kan", "kuwaa", "loo", "loona"
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

# DATA CLEANING AND BALANCING
fake_df = fake_df.dropna(subset=['Text'])
real_df = real_df.dropna(subset=['Text'])

# Ensure 'Text' column is string
fake_df["Text"] = fake_df["Text"].astype(str)
real_df["Text"] = real_df["Text"].astype(str)

print(f"Dataset Loaded: {len(fake_df)} Fake, {len(real_df)} Real")
# We do not downsample anymore to avoid losing valuable fake news features.
# We will handle the class imbalance using class_weight='balanced' in the models.

# ======================================
# PREPARE DATA
# ======================================
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
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Add extreme/vague features to TF-IDF sparse matrix
X_train_features = hstack([X_train_tfidf, np.array([ext_train, vague_train]).T])
X_test_features = hstack([X_test_tfidf, np.array([ext_test, vague_test]).T])

# ======================================
# MODELS (MUST SUPPORT predict_proba)
# ======================================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=3000, class_weight='balanced'),
    "Random_Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Naive_Bayes": MultinomialNB(alpha=0.1)
}

results = {}
trained_models = {}

print("\n===== MODEL RESULTS =====")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_features, y_train)
    preds = model.predict(X_test_features)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }
    trained_models[name] = model

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

# ======================================
# SAVE KEYWORDS FOR EXPLANATION
# Use Logistic Regression coefficients for keyword ranking
# ======================================
lr_model = trained_models["Logistic_Regression"]
feature_names = tfidf.get_feature_names_out().tolist() + ["EXT_CLAIM", "VAGUE_SRC"]
coeffs = lr_model.coef_[0]

# Positive coefs = Real, Negative coefs = Fake
feature_importance = sorted(zip(feature_names, coeffs), key=lambda x: x[1])
fake_keywords = [word for word, coef in feature_importance[:100]]
real_keywords = [word for word, coef in feature_importance[-100:]]
# Store these for app.py
keywords_dict = {"fake": fake_keywords, "real": real_keywords[::-1]}
joblib.dump(keywords_dict, "saved_model/explanation_keywords.pkl")

# ======================================
# CREATE SAVE FOLDER
# ======================================
os.makedirs("saved_model", exist_ok=True)

# Save best model based on F1-Score
best_model_name = max(results, key=lambda x: results[x]["F1-Score"])
best_model = trained_models[best_model_name]

# Fit best model on full data
X_full_tfidf = tfidf.transform(processed_texts)
X_full_features = hstack([X_full_tfidf, np.array([extreme_flags, vague_flags]).T])
best_model.fit(X_full_features, y)

# Save everything
joblib.dump(best_model, "saved_model/hybrid_model.pkl")
joblib.dump(tfidf, "saved_model/fake_real_TF_IDF_vectorizer.pkl")
joblib.dump(le, "saved_model/fake_real_label_encoder.pkl")
# Also save as the legacy name to prevent app from breaking before we update it
joblib.dump(best_model, "saved_model/svm_high_confidence.pkl")

# Save stats
stats_file = os.path.join(DATA_DIR, "stats.json")
stats = {}
if os.path.exists(stats_file):
    try:
        with open(stats_file, "r") as f:
            stats = json.load(f)
    except:
        pass

stats["model_accuracy"] = f"{results[best_model_name]['Accuracy']*100:.1f}%"
stats["model_f1"] = f"{results[best_model_name]['F1-Score']*100:.1f}%"
stats["model_precision"] = f"{results[best_model_name]['Precision']*100:.1f}%"
stats["model_recall"] = f"{results[best_model_name]['Recall']*100:.1f}%"

with open(stats_file, "w") as f:
    json.dump(stats, f, indent=4)

print(f"\nBest Model: {best_model_name}")
print(f"Hybrid model saved successfully.")

# ======================================
# ACCURACY TABLE IMAGE
# ======================================
df_results = pd.DataFrame([
    {"Model": m, **vals} for m, vals in results.items()
])
df_results = df_results.sort_values(by="F1-Score", ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Format values to 3 decimals
table_data = df_results.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    table_data[col] = table_data[col].map('{:.3f}'.format)

table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc='center',
    loc='center'
)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2E7D32")
        cell.set_text_props(color='white', weight='bold')
    else:
        cell.set_facecolor("#F1F8E9" if row % 2 == 0 else "#DCEDC8")

table.scale(1, 1.5)
plt.title("Model Performance Metrics (Upgraded System)", fontsize=14, fontweight="bold")
plt.savefig("saved_model/model_accuracy_table.png", dpi=300, bbox_inches="tight")
plt.close()

print("Full performance stats and comparison table saved in 'saved_model/'.")
print("\n[*] UPGRADE COMPLETE: TRANSITIONED TO INTELLIGENT PROBABILISTIC HYBRID SYSTEM")

