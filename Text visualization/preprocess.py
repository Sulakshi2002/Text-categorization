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
    5. Custom stopword removal
    6. Rule-based lexical normalization

Future Improvements:
    - Levenshtein distance fuzzy matching

"""


import re
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def resolve_existing_path(*relative_paths):

    for relative_path in relative_paths:

        candidate = BASE_DIR / relative_path

        if candidate.exists():

            return candidate


    return BASE_DIR / relative_paths[0]



# ============================================================
# Rule-Based Sinhala Word Normalization Dictionary
# ============================================================

NORMALIZATION_MAP = {

    # Billing
    "බිල": "බිල්පත",
    "බිල්": "බිල්පත",
    "බිල්එක": "බිල්පත",


    # Payment
    "ගෙවන්න": "ගෙවීම",
    "ගෙවීම": "ගෙවීම",


    # Internet / Network
    "නෙට්": "අන්තර්ජාලය",
    "ඉන්ටර්නෙට්": "අන්තර්ජාලය",
    "ඉන්ටනෙට්": "අන්තර්ජාලය",
    "නෙට්වර්ක්": "අන්තර්ජාලය",


    # Connectivity
    "කනෙක්ශන්": "සම්බන්ධතාව",
    "කනෙක්ෂන්": "සම්බන්ධතාව",
    "කනෙක්ට්": "සම්බන්ධතාව",
    "කනෙක්ටිවිටි": "සම්බන්ධතාව",
    "ලින්ක්": "සම්බන්ධතාව",


    # WiFi / Router
    "වයිෆායි": "වයිෆයි",
    "වයිෆයි": "වයිෆයි",
    "රවුටරේ": "රවුටර්",
    "රවුටර්": "රවුටර්",


    # Signal / Speed
    "සිග්නල්": "සංඥා",
    "ස්පීඩ්": "වේගය",
    "ස්ලෝ": "මන්දගාමී",


    # Technical issues
    "ඩැමේජ්": "හානි",
    "රිකොන්": "නැවත_සම්බන්ධ",
    "රීකනෙට්": "නැවත_සම්බන්ධ",
    "රීකනෙක්ෂන්": "නැවත_සම්බන්ධ",
    "රීස්ටාර්ට්": "නැවත_ආරම්භ",


    # Other telecom terms
    "ඩයල්ටෝන්": "ඇමතුම්_නාදය",
    "ෆයිබර්": "ෆයිබර්",
    "බ්‍රෝඩ්බෑන්ඩ්": "බ්‍රෝඩ්බෑන්ඩ්",
    "අවුට්ස්ටැන්ඩින්": "හිඟ",
    "ඩියුඩේට්": "ගෙවිය_යුතු_දිනය"

}



# ============================================================
# Additional Call Centre Stopwords
# ============================================================

EXTRA_CALLCENTER_STOPWORDS = {

    "අද",
    "එක",
    "එක්",
    "තව",
    "තවත්",
    "මෙහෙම",
    "ඔය",
    "ඕන",
    "ඕනෙ",
    "ඇයි",

    "දැන්",
    "පස්සේ",
    "පස්සෙ",

    "එතකොට",
    "මෙතන",
    "ඔතන",

    "කලින්",

    "දෙන්න",
    "ගන්න",
    "කරලා",
    "කරල",

    "නෑ",
    "නැහැ",
    "නැතුව",

    "පුළුවන්",
    "බෑ",

    "ඔව්",
    "හරි",
    "හොඳයි",

    "කියලා",
    "ගැන",

    "මට",
    "අපි",
    "අය",

    "තියෙන්නේ",
    "තියෙනවා",
    "නෑනේ",
    "නේද",
    "හිතන්නේ",
    "වෙන්න",
    "වෙනකොට",
    "කරනවද",
    "වුණාම"

}




# ============================================================
# Load Stopwords
# ============================================================

def load_stopwords(stopword_files):

    stopwords = set()


    for file in stopword_files:

        path = Path(file)


        if not path.is_absolute():

            path = BASE_DIR / path


        if not path.exists():

            print(f"Warning: {file} not found")
            continue


        with open(path, "r", encoding="utf-8") as f:

            for word in f:

                word = word.strip()

                if word:
                    stopwords.add(word)


    return stopwords




# ============================================================
# Load Transcript Files
# ============================================================

def load_transcripts(folder_path):

    transcripts = []


    folder = Path(folder_path)


    if not folder.is_absolute():

        folder = BASE_DIR / folder


    for file in folder.glob("*.txt"):

        with open(file, "r", encoding="utf-8") as f:

            text = f.read()


        transcripts.append(
            {
                "filename": file.name,
                "text": text
            }
        )


    return transcripts




# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text):


    # Unicode NFC normalization
    text = unicodedata.normalize(
        "NFC",
        text
    )


    # Convert English letters to lowercase
    text = text.lower()


    # Remove transcript language tags
    text = re.sub(
        r"\[\s*si\s*\]",
        " ",
        text
    )


    # Keep Sinhala characters only
    text = re.sub(
        r"[^\u0D80-\u0DFF\s]",
        " ",
        text
    )


    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()




# ============================================================
# Tokenization
# ============================================================

def tokenize(text):

    """
    Sinhala word-level tokenization.

    Method:
        Whitespace-based tokenization
    """

    return text.split()




# ============================================================
# Stopword Removal
# ============================================================

def remove_stopwords(tokens, stopwords):


    filtered = []


    for token in tokens:


        # Remove very short tokens
        if len(token) < 3:
            continue


        if token not in stopwords:

            filtered.append(token)


    return filtered




# ============================================================
# Rule-Based Normalization
# ============================================================

def normalize_words(tokens):


    normalized = []


    for token in tokens:


        normalized.append(
            NORMALIZATION_MAP.get(
                token,
                token
            )
        )


    return normalized




# ============================================================
# Complete Pipeline
# ============================================================

def preprocess_text(text, stopwords):


    text = clean_text(text)


    tokens = tokenize(text)


    tokens = remove_stopwords(
        tokens,
        stopwords
    )


    tokens = normalize_words(tokens)


    return tokens





# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":


    transcript_folder = resolve_existing_path(
        "transcripts",
        "trnascripts"
    )


    stopword_files = [

        resolve_existing_path("stopwords/sinhala_stopwords.txt"),

        resolve_existing_path("stopwords/callcenter_stopwords.txt")

    ]


    print("\nLoading stopwords...")


    stopwords = load_stopwords(
        stopword_files
    )


    stopwords.update(
        EXTRA_CALLCENTER_STOPWORDS
    )


    print(
        "Total stopwords:",
        len(stopwords)
    )



    print("\nLoading transcripts...")


    transcripts = load_transcripts(
        transcript_folder
    )


    print(
        "Total transcripts:",
        len(transcripts)
    )



    if transcripts:


        sample = transcripts[0]["text"]


        print("\nOriginal text:\n")

        print(sample[:300])



        processed = preprocess_text(
            sample,
            stopwords
        )


        print("\nProcessed tokens:\n")

        print(processed[:50])