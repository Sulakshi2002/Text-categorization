"""
Sinhala Word Cloud Generator
----------------------------
Builds a word cloud from any number of clean Sinhala .txt transcript files.

USAGE:
    1. Put all your cleaned .txt transcript files in one folder.
    2. Update TRANSCRIPTS_DIR and FONT_PATH below.
    3. pip install wordcloud matplotlib sentence-transformers keybert --break-system-packages
    4. python3 sinhala_wordcloud.py

Two modes are included:
    - Mode A: Fast frequency-based cloud (good default, works offline once font is downloaded)
    - Mode B: "AI/ML" mode using KeyBERT + a multilingual embedding model to score
      words/phrases by semantic importance instead of raw count (slower, needs
      internet the first time to download the model, ~470MB).
"""

import os
import re
import glob
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
TRANSCRIPTS_DIR = "./transcripts"          # folder with your .txt files
FONT_PATH = "./fonts/NotoSansSinhala-Regular.ttf"  # must be a font that supports Sinhala glyphs
STOPWORDS_FILE = "./stop_words.txt"        # one stopword per line (UTF-8) -- set to None to skip
OUTPUT_IMAGE = "./sinhala_wordcloud.png"
USE_ML_KEYWORDS = False   # set True to use Mode B (KeyBERT semantic scoring)
TOP_N = 200               # max distinct words/phrases to plot
# -----------------------------

# A small built-in fallback list, used even if STOPWORDS_FILE is missing/None.
SINHALA_STOPWORDS_BUILTIN = set("""
මම ඔබ අපි ඔවුන් ඇය ඔහු එය මට ඔබට අපට ඔවුන්ට
කරනවා තියෙනවා තිබෙනවා යනවා එනවා ඇත නැත වෙනවා
අද ඊයේ හෙට එක්ක
""".split())


def load_stopwords_from_file(path):
    """Reads a stopword list file with one word/phrase per line, ignoring
    blank lines and stray whitespace. Also strips stray leading/trailing
    zero-width joiners (U+200D) that sometimes get typed/copied in by
    accident -- these are invisible but make 'හෝ‍' != 'හෝ' as far as Python
    is concerned, which silently breaks stopword matching.

    Returns (single_words, phrases):
        single_words -- a set of one-word stopwords
        phrases      -- a list of multi-word stopword phrases (e.g. 'ඊට පස්සේ'),
                         since these can't be matched by checking one token
                         at a time -- they have to be found and removed from
                         the running text *before* it gets split into words.
    """
    if not path or not os.path.exists(path):
        print(f"(No stopwords file found at '{path}', using built-in list only.)")
        return set(), []

    single_words = set()
    phrases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = line.strip().strip("\u200d\u200c")
            if not entry:
                continue
            if " " in entry:
                phrases.append(entry)
            else:
                single_words.add(entry)

    # Remove longest phrases first so overlapping phrases don't half-match
    # (e.g. remove 'අනේ මන්දා' before a shorter phrase that could partially
    # overlap with it).
    phrases.sort(key=len, reverse=True)

    print(f"Loaded {len(single_words)} single-word stopwords and {len(phrases)} phrase stopwords from {path}")
    return single_words, phrases


# Combine your uploaded stopword file with the small built-in fallback list.
_file_single_words, SINHALA_STOPWORD_PHRASES = load_stopwords_from_file(STOPWORDS_FILE)
SINHALA_STOPWORDS = SINHALA_STOPWORDS_BUILTIN | _file_single_words


def read_all_transcripts(folder):
    texts = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    if not texts:
        raise FileNotFoundError(f"No .txt files found in {folder}")
    return texts


def clean_and_tokenize(text, stopwords, phrases=None):
    # Keep Sinhala Unicode block (U+0D80-U+0DFF) + zero-width joiner (U+200D,
    # needed for conjunct letters like ක්‍ෂ, ද්‍ය) + spaces. Strip everything else
    # (numbers, English, punctuation).
    text = re.sub(r"[^\u0D80-\u0DFF\u200D\s]", " ", text)

    # Remove multi-word stopword phrases first (e.g. 'ඊට පස්සේ') -- these
    # can't be caught by per-word filtering below since neither 'ඊට' nor
    # 'පස්සේ' alone is a stopword, only the phrase as a whole is. Punctuation
    # between the words may have collapsed into multiple spaces, so match
    # with flexible whitespace (\s+) instead of a literal single space.
    if phrases:
        for phrase in phrases:
            pattern = r"\s+".join(re.escape(w) for w in phrase.split())
            text = re.sub(pattern, " ", text)

    tokens = [
        t.strip("\u200d\u200c")
        for t in text.split()
        if t not in stopwords and t.strip("\u200d\u200c") not in stopwords and len(t) > 1
    ]
    return tokens


def mode_a_frequencies(all_text):
    tokens = clean_and_tokenize(all_text, SINHALA_STOPWORDS, SINHALA_STOPWORD_PHRASES)
    return Counter(tokens)


def mode_b_ml_keyword_scores(all_text):
    """
    Uses a multilingual sentence-embedding model via KeyBERT to score words
    by semantic relevance to the document, rather than just raw frequency.
    This tends to surface topical/meaningful words even if they don't repeat
    a lot, and downweight filler words that repeat often but carry little
    meaning.
    """
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    # Multilingual model -- supports Sinhala reasonably well since it's
    # trained on 50+ languages.
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    kw_model = KeyBERT(model=model)

    cleaned_tokens = clean_and_tokenize(all_text, SINHALA_STOPWORDS, SINHALA_STOPWORD_PHRASES)
    cleaned_text = " ".join(cleaned_tokens)

    keywords = kw_model.extract_keywords(
        cleaned_text,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=TOP_N,
        use_mmr=True,       # Maximal Marginal Relevance -> more diverse keywords
        diversity=0.6,
    )
    # keywords is a list of (phrase, score) tuples; scale scores up for the
    # word cloud (it just needs relative weights).
    return {phrase: score * 100 for phrase, score in keywords}


def main():
    texts = read_all_transcripts(TRANSCRIPTS_DIR)
    all_text = "\n".join(texts)
    print(f"Loaded {len(texts)} transcript file(s), {len(all_text)} characters total.")

    if USE_ML_KEYWORDS:
        print("Scoring keywords with multilingual embeddings (KeyBERT)...")
        freqs = mode_b_ml_keyword_scores(all_text)
    else:
        print("Counting word frequencies...")
        freqs = mode_a_frequencies(all_text)

    if not freqs:
        raise ValueError("No words survived cleaning/stopword removal -- check your stopword list or input text.")

    top = Counter(freqs).most_common(15)
    print("Top terms:", top)

    wc = WordCloud(
        font_path=FONT_PATH,   # REQUIRED for Sinhala glyphs to render (not boxes)
        width=1600,
        height=900,
        background_color="white",
        collocations=False,   # we already control phrasing ourselves
        max_words=TOP_N,
    ).generate_from_frequencies(dict(freqs))

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_IMAGE, dpi=200, bbox_inches="tight")
    print(f"Saved word cloud to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
