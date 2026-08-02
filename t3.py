from collections import Counter
import glob
import json
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Resolves the exact directory where t3.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KEYWORDS_DIR = os.path.join(BASE_DIR, "keywords")
TRANSCRIPTS_DIR = os.path.join(
    BASE_DIR, "transcriptions"
)  # Matches your sidebar folder

# Files located directly in your main testing directory
FONT_PATH = os.path.join(BASE_DIR, "NotoSansSinhala-Regular.ttf")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "sinhala_slt_wordcloud.png")
OUTPUT_CSV = os.path.join(BASE_DIR, "matched_keywords_summary.csv")

# Scoring Mode:
#   "freq_times_weight" -> Size = (occurrences in transcript) x (JSON weight) [Recommended]
#   "freq_only"          -> Size = occurrences in transcript (JSON acts as filter only)
#   "weight_only"        -> Size = JSON weight only
SCORE_MODE = "freq_times_weight"
TOP_N_WORDS = 200


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def normalize_sinhala(text):
    """Deep normalization for Sinhala Unicode string matching.

    Strips hidden Zero-Width Joiners (ZWJ \u200D) and Zero-Width Non-Joiners
    (\u200C) both from edges and internally to prevent hidden string mismatches.
    """
    if not text:
        return ""
    text = text.replace("\u200d", "").replace("\u200c", "")
    return text.strip()


def load_and_merge_keyword_jsons(folder_path):
    """Loads all category JSON files from folder_path, resolves duplicate word

    collisions by keeping the highest weight, and returns a merged dictionary.
    """
    merged_keywords = {}
    collisions = []

    json_paths = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    if not json_paths:
        raise FileNotFoundError(
            f"No .json keyword files found in directory: '{folder_path}'"
        )

    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for raw_word, weight in data.items():
                    word = normalize_sinhala(raw_word)
                    if not word:
                        continue
                    if word in merged_keywords and merged_keywords[word] != weight:
                        collisions.append((
                            word,
                            merged_keywords[word],
                            weight,
                            os.path.basename(path),
                        ))
                    merged_keywords[word] = max(
                        merged_keywords.get(word, 0.0), float(weight)
                    )
        except Exception as e:
            print(f"[Warning] Failed to parse JSON file {path}: {e}")

    print(
        f"Successfully loaded {len(json_paths)} JSON file(s) -> {len(merged_keywords)} unique keywords merged."
    )
    if collisions:
        print(
            f"Note: {len(collisions)} overlapping words detected across category files. Kept highest weight."
        )

    return merged_keywords


def read_transcripts(folder_path):
    """Reads all .txt transcript files in folder_path with UTF-8 encoding."""
    texts = []
    txt_paths = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

    if not txt_paths:
        raise FileNotFoundError(
            f"No .txt transcript files found in directory: '{folder_path}'"
        )

    for path in txt_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    texts.append(content)
        except Exception as e:
            print(f"[Warning] Failed to read transcript {path}: {e}")

    print(f"Successfully loaded {len(texts)} transcript file(s).")
    return "\n".join(texts)


def tokenize_sinhala(text):
    """Tokenizes text preserving only Sinhala Unicode characters (\u0D80-\u0DFF)."""
    clean_text = re.sub(r"[^\u0D80-\u0DFF\s]", " ", text)
    raw_tokens = clean_text.split()

    normalized_tokens = []
    for token in raw_tokens:
        norm = normalize_sinhala(token)
        if len(norm) > 1:
            normalized_tokens.append(norm)

    return normalized_tokens


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    # 1. Load Dictionary & Transcripts
    keyword_dict = load_and_merge_keyword_jsons(KEYWORDS_DIR)
    full_transcript_text = read_transcripts(TRANSCRIPTS_DIR)

    # 2. Tokenize and calculate raw transcript frequency
    tokens = tokenize_sinhala(full_transcript_text)
    raw_counts = Counter(tokens)

    # 3. Filter strictly against merged keyword whitelist
    matched_data = {}
    for word, count in raw_counts.items():
        if word in keyword_dict:
            weight = keyword_dict[word]

            if SCORE_MODE == "freq_times_weight":
                score = count * weight
            elif SCORE_MODE == "weight_only":
                score = weight
            else:  # "freq_only"
                score = count

            matched_data[word] = {
                "count": count,
                "weight": weight,
                "final_score": score,
            }

    if not matched_data:
        raise ValueError(
            "No matching keywords were found between the transcripts and your JSON dictionaries. "
            "Please check word spellings or transcript content."
        )

    print(
        f"Matched {len(matched_data)} unique domain keywords out of {len(keyword_dict)} whitelist terms."
    )

    # 4. Generate Pandas DataFrame & Export CSV Summary
    df_summary = pd.DataFrame([
        {
            "Keyword": word,
            "Transcript_Frequency": meta["count"],
            "JSON_Weight": meta["weight"],
            "Final_Score": meta["final_score"],
        }
        for word, meta in matched_data.items()
    ]).sort_values(by="Final_Score", ascending=False)

    df_summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved frequency and score analysis CSV to '{OUTPUT_CSV}'.")

    print("\n=== TOP 10 MATCHED KEYWORDS ===")
    print(df_summary.head(10).to_string(index=False))

    # 5. Extract frequency mapping for WordCloud generator
    wordcloud_frequencies = {
        word: meta["final_score"] for word, meta in matched_data.items()
    }

    # 6. Render & Save Word Cloud
    if not os.path.exists(FONT_PATH):
        print(
            f"\n[Warning] Font file '{FONT_PATH}' not found in main directory! "
            "Please ensure 'NotoSansSinhala-Regular.ttf' is inside your testing folder."
        )

    wc = WordCloud(
        font_path=FONT_PATH,
        width=1600,
        height=900,
        background_color="white",
        collocations=False,
        max_words=TOP_N_WORDS,
        colormap="Dark2",
    ).generate_from_frequencies(wordcloud_frequencies)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(
        f"SLT Domain Keywords WordCloud (Scoring Mode: {SCORE_MODE})",
        fontsize=18,
        pad=20,
    )
    plt.tight_layout(pad=0)

    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    print(
        f"\nSuccessfully generated and saved high-res WordCloud to '{OUTPUT_IMAGE}'."
    )


if __name__ == "__main__":
    main()