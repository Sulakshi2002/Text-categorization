import sys
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH = r"D:\SLT\New folder-01\Transcription for cleaning\data\transcripts\full_transcripts.csv"
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------
# Load targeted category dictionaries
# ---------------------------------------------------
def load_dictionary(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

bill_dict = load_dictionary(BASE_DIR / "dictionary" / "Bill Inquries.txt")
technical_dict = load_dictionary(BASE_DIR / "dictionary" / "Fault and technical assistance.txt")
product_dict = load_dictionary(BASE_DIR / "dictionary" / "New Product.txt")

ALL_STEMS = list(set(bill_dict + technical_dict + product_dict))

# =====================================================================
# TRUE LEVENSHTEIN DISTANCE MATRIX ALGORITHM
# =====================================================================
def levenshtein_similarity(s1, s2):
    """
    Calculates the exact Levenshtein Edit Distance matrix.
    Converts it to a normalized similarity ratio between 0.0 and 1.0.
    """
    m, n = len(s1), len(s2)
    
    # Base cases for empty strings
    if m == 0: return 0.0 if n > 0 else 1.0
    if n == 0: return 0.0
    
    # Initialize the cost tracking row matrix
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            # If the characters match exactly, transformation cost is 0
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            
            # Find the minimum cost path among Deletion, Insertion, and Substitution
            curr_row[j] = min(
                prev_row[j] + 1,       # Deletion
                curr_row[j - 1] + 1,  # Insertion
                prev_row[j - 1] + cost # Substitution
            )
        prev_row = list(curr_row)
        
    edit_distance = curr_row[n]
    max_len = max(m, n)
    
    # Normalize the edit cost into an accurate similarity ratio metric (0.0 to 1.0)
    return 1.0 - (edit_distance / max_len)

# =====================================================================
# HIGH-SPEED MULTI-TIERED CASCADING TOKENIZER ENGINE
# =====================================================================
def preprocess_and_tokenize_text(raw_text):
    """
    Optimized multi-tier cleaner using True Levenshtein Distance.
    Processes a single text string into a list of tokens.
    """
    raw_text = raw_text.lower()
    clean_text = re.sub(r'[^\u0D80-\u0DFF\s]', '', raw_text)
    raw_words = clean_text.split()
    
    final_tokens = []
    
    for word in raw_words:
        # Tier 1: Exact Match
        if word in ALL_STEMS:
            final_tokens.append(word)
            continue
            
        # Tier 2: Substring Match
        matched = False
        for stem in ALL_STEMS:
            if stem in word:
                final_tokens.append(stem)
                suffix = word.replace(stem, "").strip()
                if suffix:
                    final_tokens.append(suffix)
                matched = True
                break
                
        if matched:
            continue
            
        # Tier 3: True Levenshtein Distance Matrix Matching
        for stem in ALL_STEMS:
            # SPEED PATCH: Skip matrix math if string lengths vary too widely
            if abs(len(word) - len(stem)) > 2:
                continue
                
            raw_segment = word[:len(stem)]
            
            # Execute true Levenshtein distance check instead of SequenceMatcher
            similarity = levenshtein_similarity(stem, raw_segment)
            
            # 0.85 threshold captures single-vowel changes or missing virama characters (්)
            if similarity >= 0.85:
                final_tokens.append(stem)
                matched = True
                break
                
        if not matched:
            final_tokens.append(word)
            
    return final_tokens

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
df = pd.read_csv(CSV_PATH, dtype=str)
df = df.dropna(subset=["transcript", "label"])

X_raw = df["transcript"].values
y = df["label"].values

print(f"Total usable rows: {len(df)}")
print(f"Category distribution:\n{df['label'].value_counts()}\n")

# =====================================================================
# SPEED OPTIMIZATION: TOKENIZE CORPUS ONCE
# =====================================================================
print("Pre-tokenizing corpus using True Levenshtein logic...")
X_tokenized_strings = []
for text in X_raw:
    tokens = preprocess_and_tokenize_text(text)
    X_tokenized_strings.append(" ".join(tokens))
X = np.array(X_tokenized_strings)
print("Pre-tokenization complete. Launching validation pipeline.\n")

loo = LeaveOneOut()
y_true_all = []
y_pred_all = []

fold_count = 0
total_folds = len(X)

for train_idx, test_idx in loo.split(X):
    fold_count += 1
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Clean space tokenization configuration
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=str.split, 
        token_pattern=None,
        lowercase=False, 
        ngram_range=(1, 2),
        min_df=1
    )
    
    X_train_final = vectorizer.fit_transform(X_train)
    X_test_final = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=0.3,                    
        class_weight='balanced',
        max_iter=11000,
    )
    model.fit(X_train_final, y_train)
    pred = model.predict(X_test_final)

    y_true_all.append(y_test[0])
    y_pred_all.append(pred[0])

    if fold_count % 10 == 0:
        print(f"  ...processed {fold_count}/{total_folds} folds")

# ---------------------------------------------------
# Results
# ---------------------------------------------------
acc = accuracy_score(y_true_all, y_pred_all)
print("=" * 60)
print(f"LOGISTIC REGRESSION - TRUE LEVENSHTEIN ACCURACY: {acc*100:.1f}% ({sum(t==p for t,p in zip(y_true_all,y_pred_all))}/{len(y_true_all)})")
print("=" * 60)

print("\n--- Classification Report ---")
print(classification_report(y_true_all, y_pred_all, zero_division=0))

print("\n--- Confusion Matrix ---")
labels = sorted(set(y_true_all))
cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
print("Labels order:", labels)
print(cm)

print("\n--- Misclassified rows ---")
for i, (true, pred) in enumerate(zip(y_true_all, y_pred_all)):
    if true != pred:
        # Get the actual raw transcript since there is no Audio_file column
        raw_text = df.iloc[i]["transcript"]
        
        # Truncate long sentences to 60 characters so it fits neatly in your terminal window
        short_text = raw_text[:60] + "..." if len(raw_text) > 60 else raw_text
        
        # df.index[i] provides the row number (0 to 57) from your CSV file
        print(f"Row {df.index[i]}: true={true} | predicted={pred} ")

