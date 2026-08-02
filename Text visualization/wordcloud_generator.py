"""
Sinhala Call Centre Word Cloud Generator

Purpose:
    Generate word cloud using cleaned Sinhala tokens
"""


from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocess import (
    load_stopwords,
    load_transcripts,
    preprocess_text
)


BASE_DIR = Path(__file__).resolve().parent


def resolve_existing_path(*relative_paths):

    for relative_path in relative_paths:

        candidate = Path(relative_path)

        if not candidate.is_absolute():

            candidate = BASE_DIR / candidate

        if candidate.exists():

            return candidate


    return None



# ============================================================
# Configuration
# ============================================================

TRANSCRIPT_FOLDER = BASE_DIR / "transcripts"

STOPWORD_FILES = [

    BASE_DIR / "stopwords/sinhala_stopwords.txt",

    BASE_DIR / "stopwords/callcenter_stopwords.txt"

]


FONT_PATH = resolve_existing_path(
    "fonts/NotoSansSinhala-Regular.ttf",
    r"C:\Windows\Fonts\Nirmala.ttc",
    r"C:\Windows\Fonts\DejaVuSans.ttf",
)


if FONT_PATH is None:

    raise FileNotFoundError(
        "No usable font found. Add fonts/NotoSansSinhala-Regular.ttf or install a Sinhala-capable Windows font."
    )


OUTPUT_FILE = BASE_DIR / "outputs" / "sinhala_wordcloud.png"



# ============================================================
# Load stopwords
# ============================================================

print("Loading stopwords...")

stopwords = load_stopwords(
    STOPWORD_FILES
)


print(
    "Total stopwords:",
    len(stopwords)
)



# ============================================================
# Load transcripts
# ============================================================

print("\nLoading transcripts...")


transcripts = load_transcripts(
    TRANSCRIPT_FOLDER
)


print(
    "Total transcripts:",
    len(transcripts)
)



# ============================================================
# Preprocess all transcripts
# ============================================================

all_tokens = []


for transcript in transcripts:


    tokens = preprocess_text(
        transcript["text"],
        stopwords
    )


    all_tokens.extend(tokens)



print(
    "\nTotal tokens after preprocessing:",
    len(all_tokens)
)



# ============================================================
# Word Frequency
# ============================================================

frequency = Counter(all_tokens)


print("\nTop 20 words:")

for word, count in frequency.most_common(20):

    print(
        word,
        ":",
        count
    )



# ============================================================
# Generate Word Cloud
# ============================================================

text = " ".join(all_tokens)



wordcloud = WordCloud(

    font_path=FONT_PATH,

    width=1200,

    height=800,

    background_color="white",

    max_words=100,

    collocations=False

).generate(text)



# ============================================================
# Display
# ============================================================


plt.figure(
    figsize=(12,8)
)


plt.imshow(
    wordcloud,
    interpolation="bilinear"
)


plt.axis("off")


plt.title(
    "Sinhala Call Centre Word Cloud"
)


plt.show()



# ============================================================
# Save output
# ============================================================

Path("outputs").mkdir(
    exist_ok=True
)


wordcloud.to_file(
    OUTPUT_FILE
)


print(
    "\nSaved:",
    OUTPUT_FILE
)