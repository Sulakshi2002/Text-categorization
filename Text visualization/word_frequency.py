"""
Sinhala Call Centre Word Frequency Generator

Purpose:
    Generate word frequency statistics
    after Sinhala text preprocessing.

Pipeline:

Raw Transcripts
        |
        v
preprocess.py
        |
        v
Clean Tokens
        |
        v
Frequency Counting
        |
        v
CSV Output

"""


from collections import Counter
from pathlib import Path
import csv

from preprocess import (
    load_transcripts,
    load_stopwords,
    preprocess_text,
    resolve_existing_path
)



# ============================================================
# Configuration
# ============================================================


# Use only 5 transcripts for testing
SAMPLE_SIZE = 5


OUTPUT_FILE = Path("outputs/word_frequency.csv")



# ============================================================
# Main Processing
# ============================================================


if __name__ == "__main__":


    print("\nLoading stopwords...")


    stopword_files = [

        resolve_existing_path(
            "stopwords/sinhala_stopwords.txt"
        )

    ]


    stopwords = load_stopwords(
        stopword_files
    )


    print(
        "Total stopwords:",
        len(stopwords)
    )



    print("\nLoading transcripts...")


    transcript_folder = resolve_existing_path(
        "transcripts"
    )


    transcripts = load_transcripts(
        transcript_folder
    )


    print(
        "Total transcripts:",
        len(transcripts)
    )



    # Select only first 5 transcripts

    transcripts = transcripts[:SAMPLE_SIZE]


    print(
        "Processing transcripts:",
        len(transcripts)
    )



    all_tokens = []



    for transcript in transcripts:


        processed_tokens = preprocess_text(

            transcript["text"],

            stopwords

        )


        all_tokens.extend(
            processed_tokens
        )



    print(
        "\nTotal tokens after preprocessing:",
        len(all_tokens)
    )



    # ========================================================
    # Word Frequency Calculation
    # ========================================================


    word_counts = Counter(
        all_tokens
    )



    print("\nTop 20 Words:\n")


    for word, count in word_counts.most_common(20):

        print(
            f"{word} : {count}"
        )



    # ========================================================
    # Save CSV
    # ========================================================


    OUTPUT_FILE.parent.mkdir(
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:


        writer = csv.writer(file)


        writer.writerow(
            [
                "Word",
                "Frequency"
            ]
        )


        for word, frequency in word_counts.most_common():


            writer.writerow(
                [
                    word,
                    frequency
                ]
            )



    print(
        "\nWord frequency saved:"
    )


    print(
        OUTPUT_FILE
    )