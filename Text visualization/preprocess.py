"""
Sinhala Call Centre Text Preprocessing Module

Purpose:
    Prepare Sinhala transcripts before generating
    word clouds and keyword visualizations.

Preprocessing Steps:
    1. Load transcripts
    2. Unicode NFC normalization
    3. Text cleaning
    4. Sinhala word-level tokenization
    5. Word normalization against canonical terms
    6. Stopword removal
"""


import re
import unicodedata
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CANONICAL_TERMS_FILE = "dictionaries/canonical_terms.txt"


def resolve_existing_path(*relative_paths):

    for relative_path in relative_paths:

        candidate = BASE_DIR / relative_path

        if candidate.exists():

            return candidate


    return BASE_DIR / relative_paths[0]


SHORT_WORD_WHITELIST = {
    "බිල": "බිල්පත",
    "නෙට්": "ඉන්ටර්නෙට්",
    "පේ": "පේමන්ට්",
    "ඩියු": "ඩියුඩේට්",
}

SINHALA_VOWEL_MARKS = re.compile(r"[ාැෑිීුූෘෲෙේෛොෝෞ්‍්]")

SKELETON_TRANSLATION = str.maketrans(
    {
        "ඛ": "ක",
        "ඝ": "ක",
        "ඟ": "ග",
        "ඡ": "ච",
        "ජ": "ච",
        "ඣ": "ච",
        "ඤ": "න",
        "ඪ": "ට",
        "ඩ": "ට",
        "ඬ": "ට",
        "ණ": "න",
        "ථ": "ත",
        "ධ": "ද",
        "ඳ": "ද",
        "බ": "ප",
        "භ": "ප",
        "ඵ": "ප",
        "ශ": "ස",
        "ෂ": "ස",
        "ළ": "ල",
        "ෆ": "ප",
    }
)

PHONETIC_TRANSLATION = str.maketrans(
    {
        "ඛ": "ක",
        "ඝ": "ක",
        "ඟ": "ග",
        "ඡ": "ච",
        "ජ": "ජ",
        "ඣ": "ජ",
        "ඤ": "ඤ",
        "ඪ": "ඩ",
        "ඩ": "ඩ",
        "ඬ": "ඩ",
        "ණ": "න",
        "ථ": "ත",
        "ධ": "ද",
        "ඳ": "ද",
        "බ": "බ",
        "භ": "බ",
        "ඵ": "ප",
        "ශ": "ස",
        "ෂ": "ස",
        "ළ": "ල",
        "ෆ": "ෆ",
    }
)


def load_canonical_terms(file_path):

    canonical_terms = []
    seen = set()


    path = resolve_existing_path(file_path)


    if not path.exists():

        print(f"Warning: canonical terms file not found: {path}")

        return canonical_terms


    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            term = line.strip()


            if not term or term.startswith("#"):

                continue


            if term not in seen:

                canonical_terms.append(term)
                seen.add(term)


    return canonical_terms


CANONICAL_TERMS = tuple(load_canonical_terms(CANONICAL_TERMS_FILE))
CANONICAL_SET = set(CANONICAL_TERMS)
SHORT_CANONICAL_TERMS = tuple(term for term in CANONICAL_TERMS if len(term) <= 3)
MEDIUM_CANONICAL_TERMS = tuple(term for term in CANONICAL_TERMS if 4 <= len(term) <= 5)
LONG_CANONICAL_TERMS = tuple(term for term in CANONICAL_TERMS if len(term) >= 6)


def load_stopwords(stopword_files):

    stopwords = set()


    for file in stopword_files:

        path = Path(file)


        if not path.is_absolute():

            path = BASE_DIR / path


        if not path.exists():

            print(f"Warning: {file} not found")

            continue


        with open(path, "r", encoding="utf-8") as handle:

            for word in handle:

                word = word.strip()


                if word:

                    stopwords.add(word)


    return stopwords


def load_transcripts(folder_path):

    transcripts = []


    folder = Path(folder_path)


    if not folder.is_absolute():

        folder = BASE_DIR / folder


    for file in folder.glob("*.txt"):

        with open(file, "r", encoding="utf-8") as handle:

            text = handle.read()


        transcripts.append(
            {
                "filename": file.name,
                "text": text,
            }
        )


    return transcripts


def clean_text(text):

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\[\s*si\s*\]", " ", text)
    text = re.sub(r"[^\u0D80-\u0DFF\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    """Sinhala word-level tokenization."""

    return re.findall(r"[\u0D80-\u0DFF]+", text)


def remove_stopwords(tokens, stopwords):

    filtered = []


    for token in tokens:

        if len(token) < 3:

            continue


        if token not in stopwords:

            filtered.append(token)


    return filtered


@lru_cache(maxsize=None)
def _levenshtein_distance(left, right):

    if left == right:

        return 0


    if not left:

        return len(right)


    if not right:

        return len(left)


    if len(left) < len(right):

        left, right = right, left


    previous_row = list(range(len(right) + 1))


    for left_index, left_char in enumerate(left, start=1):

        current_row = [left_index]


        for right_index, right_char in enumerate(right, start=1):

            insertion_cost = current_row[right_index - 1] + 1
            deletion_cost = previous_row[right_index] + 1
            substitution_cost = previous_row[right_index - 1]


            if left_char != right_char:

                substitution_cost += 1


            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))


        previous_row = current_row


    return previous_row[-1]


def _normalized_similarity(left, right):

    if not left and not right:

        return 1.0


    max_length = max(len(left), len(right), 1)

    return 1.0 - (_levenshtein_distance(left, right) / max_length)


@lru_cache(maxsize=None)
def _strip_vowel_marks(token):

    return SINHALA_VOWEL_MARKS.sub("", token)


@lru_cache(maxsize=None)
def _consonant_skeleton(token):

    return _strip_vowel_marks(token).translate(SKELETON_TRANSLATION)


@lru_cache(maxsize=None)
def _phonetic_signature(token):

    return _strip_vowel_marks(token).translate(PHONETIC_TRANSLATION)


def _select_medium_candidate(token):

    best_candidate = token
    best_score = 0.0


    for candidate in MEDIUM_CANONICAL_TERMS:

        score = _normalized_similarity(token, candidate)


        if score > best_score:

            best_score = score
            best_candidate = candidate


    if best_score >= 0.75:
        return best_candidate

    return token

def _select_long_candidate(token):

    best_candidate = token
    best_score = -1.0


    for candidate in LONG_CANONICAL_TERMS:

        skeleton_score = _normalized_similarity(
            _consonant_skeleton(token),
            _consonant_skeleton(candidate)
        )
        phonetic_score = _normalized_similarity(
            _phonetic_signature(token),
            _phonetic_signature(candidate)
        )
        levenshtein_score = _normalized_similarity(
            _strip_vowel_marks(token),
            _strip_vowel_marks(candidate)
        )

        score = (
            0.40 * skeleton_score
            + 0.30 * phonetic_score
            + 0.30 * levenshtein_score
        )


        if score > best_score:

            best_score = score
            best_candidate = candidate


    if best_score >= 0.72:
        return best_candidate

    return token


def normalize_words(tokens):

    normalized = []


    for token in tokens:

        if token in CANONICAL_SET:

            normalized.append(token)

            continue


        if len(token) <= 3:

            normalized.append(
                SHORT_WORD_WHITELIST.get(token, token)
            )

            continue


        if 4 <= len(token) <= 5:

            normalized.append(_select_medium_candidate(token))

            continue


        new_word = _select_long_candidate(token)

        if token != new_word:
            print(token, "---->", new_word)

        normalized.append(new_word)


    return normalized


def preprocess_text(text, stopwords):

    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = normalize_words(tokens)
    

    return tokens


if __name__ == "__main__":

    transcript_folder = resolve_existing_path(
        "transcripts",
        "trnascripts"
    )

    stopword_files = [
        resolve_existing_path("stopwords/sinhala_stopwords.txt")
    ]

    print("\nLoading stopwords...")

    stopwords = load_stopwords(stopword_files)

    print("Total stopwords:", len(stopwords))

    print("\nLoading transcripts...")

    transcripts = load_transcripts(transcript_folder)

    print("Total transcripts:", len(transcripts))

    if transcripts:

        sample = transcripts[0]["text"]

        print("\nOriginal text:\n")
        print(sample[:300])

        processed = preprocess_text(sample, stopwords)

        print("\nProcessed tokens:\n")
        print(processed[:50])