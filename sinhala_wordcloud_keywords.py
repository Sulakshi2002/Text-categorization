"""
Sinhala Word Cloud - Keyword-Filtered (Combined Across Categories)
-------------------------------------------------------------------
You have 4 category JSON files (bill_inquiries, fault_and_technical,
product_and_new_service, telephone_number_request_or_other), each mapping
Sinhala words -> a relevance weight (0-1).

This script:
    1. Merges all 4 JSON files into ONE combined keyword whitelist
       (if the same word appears in more than one file with different
       weights, the HIGHEST weight wins).
    2. Reads your transcripts and counts how often each word actually
       occurs.
    3. Keeps ONLY words that are in that combined whitelist -- every other
       word in the transcripts is ignored completely.
    4. Produces ONE single word cloud combining all 4 categories together
       (NOT 4 separate clouds).
    5. Sizes each word by: (how often it appears in your transcripts) x
       (its relevance weight from the JSON). So a word that's both common
       AND marked highly relevant ends up biggest.

USAGE:
    1. Put your 4 (or however many) category JSON files in ./keywords/
    2. Put your transcript .txt files in ./transcripts/
    3. pip install wordcloud matplotlib
    4. python sinhala_wordcloud_keywords.py
"""

import os
import re
import json
import glob
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
KEYWORDS_DIR = "./keywords"                # folder containing your category .json files
TRANSCRIPTS_DIR = "./transcripts"          # folder with your .txt transcript files
FONT_PATH = "./fonts/NotoSansSinhala-Regular.ttf"
OUTPUT_IMAGE = "./sinhala_keyword_wordcloud.png"

# Scoring mode:
#   "freq_times_weight" -> size = (times seen in transcripts) x (JSON weight)   [default, recommended]
#   "freq_only"          -> size = (times seen in transcripts) -- JSON only used as a whitelist filter
#   "weight_only"        -> size = JSON weight only -- ignores how often it actually appears
SCORE_MODE = "freq_times_weight"
TOP_N = 200
# -----------------------------


def _normalize(word):
    """Strip stray zero-width joiners so hidden-character variants of the
    same visible word still match (see earlier fix -- same issue as with
    stopwords)."""
    return word.strip().strip("\u200d\u200c")


def load_and_merge_keyword_jsons(folder):
    """Loads every .json file in folder, merges them into one dict. If the
    same word appears in multiple files with different weights, keeps the
    highest weight and reports the collision so you can see it happened."""
    merged = {}
    collisions = []

    json_paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in {folder}")

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for raw_word, weight in data.items():
            word = _normalize(raw_word)
            if word in merged and merged[word] != weight:
                collisions.append((word, merged[word], weight, os.path.basename(path)))
            merged[word] = max(merged.get(word, 0), weight)

    print(f"Merged {len(json_paths)} keyword files -> {len(merged)} unique words total.")
    if collisions:
        print(f"({len(collisions)} words appeared in more than one file with different weights -- kept the highest.)")

    return merged


def read_all_transcripts(folder):
    texts = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    if not texts:
        raise FileNotFoundError(f"No .txt files found in {folder}")
    return texts


def tokenize(text):
    # Keep Sinhala Unicode block + zero-width joiner (needed for conjunct
    # letters) + spaces. Strip everything else (numbers, English, punctuation).
    text = re.sub(r"[^\u0D80-\u0DFF\u200D\s]", " ", text)
    return [_normalize(t) for t in text.split() if len(t) > 1]


def main():
    keyword_weights = load_and_merge_keyword_jsons(KEYWORDS_DIR)

    texts = read_all_transcripts(TRANSCRIPTS_DIR)
    all_text = "\n".join(texts)
    print(f"Loaded {len(texts)} transcript file(s), {len(all_text)} characters total.")

    tokens = tokenize(all_text)
    transcript_freq = Counter(tokens)

    # Keep ONLY words that exist in the merged keyword whitelist.
    matched = {w: c for w, c in transcript_freq.items() if w in keyword_weights}
    print(f"{len(matched)} of the {len(keyword_weights)} keyword-list words were actually found in your transcripts.")

    if not matched:
        raise ValueError(
            "None of your keyword-list words were found in the transcripts. "
            "Double-check TRANSCRIPTS_DIR and that these are the right transcripts for this vocabulary."
        )

    if SCORE_MODE == "freq_times_weight":
        scores = {w: matched[w] * keyword_weights[w] for w in matched}
    elif SCORE_MODE == "weight_only":
        scores = {w: keyword_weights[w] for w in matched}
    else:  # freq_only
        scores = dict(matched)

    top = Counter(scores).most_common(15)
    print("Top terms:", top)

    wc = WordCloud(
        font_path=FONT_PATH,
        width=1600,
        height=900,
        background_color="white",
        collocations=False,
        max_words=TOP_N,
    ).generate_from_frequencies(scores)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_IMAGE, dpi=200, bbox_inches="tight")
    print(f"Saved combined keyword word cloud to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
