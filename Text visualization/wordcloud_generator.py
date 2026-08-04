"""
Sinhala Call Centre Word Cloud Generator

Purpose:
    Generate Sinhala word cloud using
    pre-calculated word frequencies.

Input:
    outputs/word_frequency.csv

Output:
    outputs/sinhala_wordcloud.png

Pipeline:

word_frequency.csv
        |
        ↓
Load frequencies
        |
        ↓
Generate Word Cloud
        |
        ↓
Save image

"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud



# ============================================================
# Configuration
# ============================================================


BASE_DIR = Path(__file__).resolve().parent


def resolve_existing_path(*relative_paths):

    for relative_path in relative_paths:

        candidate = BASE_DIR / relative_path

        if candidate.exists():

            return candidate


    return BASE_DIR / relative_paths[0]


FREQUENCY_FILE = (
    BASE_DIR /
    "outputs" /
    "word_frequency.csv"
)


FONT_PATH = (
    resolve_existing_path(
        "fonts/Noto_Sans_Sinhala/static/NotoSansSinhala_SemiCondensed-Regular.ttf",
        "fonts/Noto_Sans_Sinhala/NotoSansSinhala-VariableFont_wdth,wght.ttf",
        "fonts/NotoSansSinhala-Regular.ttf"
    )
)


OUTPUT_FILE = (
    BASE_DIR /
    "outputs" /
    "sinhala_wordcloud.png"
)



# ============================================================
# Load Word Frequencies
# ============================================================


def load_word_frequency(file_path):

    """
    Load word frequency CSV.

    Expected format:

    Word,Frequency

    """


    df = pd.read_csv(
        file_path,
        encoding="utf-8"
    )


    frequency_dict = dict(
        zip(
            df["Word"],
            df["Frequency"]
        )
    )


    return frequency_dict



# ============================================================
# Generate Word Cloud
# ============================================================


def generate_wordcloud(frequency_dict):


    wordcloud = WordCloud(

        font_path=str(FONT_PATH),

        width=1200,

        height=800,

        background_color="white",

        max_words=100,

        min_font_size=10,

        max_font_size=120,

        prefer_horizontal=0.9,

    ).generate_from_frequencies(
        frequency_dict
    )


    return wordcloud




# ============================================================
# Save Visualization
# ============================================================


def save_wordcloud(wordcloud, output_path):


    plt.figure(
        figsize=(12,8)
    )


    plt.imshow(
        wordcloud,
        interpolation="bilinear"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()




# ============================================================
# Main Execution
# ============================================================


if __name__ == "__main__":


    print("\nLoading word frequency...")


    if not FREQUENCY_FILE.exists():

        raise FileNotFoundError(

            f"Frequency file not found: {FREQUENCY_FILE}"

        )



    if not FONT_PATH.exists():

        raise FileNotFoundError(

            f"Sinhala font not found: {FONT_PATH}"

        )



    frequencies = load_word_frequency(
        FREQUENCY_FILE
    )


    print(
        "Total words:",
        len(frequencies)
    )



    print("\nGenerating word cloud...")


    wc = generate_wordcloud(
        frequencies
    )



    OUTPUT_FILE.parent.mkdir(
        exist_ok=True
    )



    save_wordcloud(

        wc,

        OUTPUT_FILE

    )



    print(
        "\nWord cloud saved:"
    )


    print(
        OUTPUT_FILE
    )