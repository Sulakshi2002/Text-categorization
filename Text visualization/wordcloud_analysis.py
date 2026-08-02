"""
Sinhala Telecom Transcript Category Word Cloud Generator

Author: NLP/Data Visualization Engineer
Purpose:
    Generate category-colored Sinhala word clouds from
    call center transcript tokens.
"""


import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter
from matplotlib.patches import Patch

from classifier import SinhalaTelecomClassifier



# ============================================================
# Category Color Configuration
# ============================================================

CATEGORY_COLORS = {

    "Bill_Inquiries": "#e66101",              # Orange

    "Fault_and_Technical": "#5e3c99",         # Purple

    "Product_And_New_Service": "#2ca25f",     # Green

    "Telephone_Number_Request_Or_Other": "#999999"  # Grey
}



DEFAULT_COLOR = "#333333"



# ============================================================
# Category Color Function
# ============================================================

class SinhalaCategoryColorMapper:
    """
    Maps each Sinhala word to a category color
    using classifier dictionary patterns.
    """


    def __init__(
        self,
        classifier: SinhalaTelecomClassifier,
        similarity_threshold: float = 0.80
    ):

        self.classifier = classifier
        self.threshold = similarity_threshold



    def get_category(self, word: str):
        """
        Identify the category of a Sinhala token.

        Matching process:
            1. Compare token against category dictionaries.
            2. Use classifier Levenshtein similarity.
            3. Return highest matching category.
        """

        best_category = None
        best_score = 0


        for category, patterns in self.classifier.domain_patterns.items():

            for pattern in patterns:

                score = self.classifier._calculate_similarity(
                    word,
                    pattern
                )


                if score > best_score:

                    best_score = score
                    best_category = category



        if best_score >= self.threshold:

            return best_category


        return None



    def color_func(
        self,
        word,
        font_size,
        position,
        orientation,
        random_state=None,
        **kwargs
    ):
        """
        WordCloud callback function.

        Returns hex color based on detected category.
        """


        category = self.get_category(word)


        if category in CATEGORY_COLORS:

            return CATEGORY_COLORS[category]


        return DEFAULT_COLOR



# ============================================================
# Word Cloud Generator
# ============================================================


def generate_sinhala_wordcloud(
    frequencies,
    classifier,
    font_path,
    output_file=None
):
    """
    Generate Sinhala category-coded word cloud.

    Parameters
    ----------
    frequencies : dict or Counter
        Cleaned Sinhala token frequencies.

    classifier :
        SinhalaTelecomClassifier instance.

    font_path :
        Path to Sinhala TTF font.

    output_file :
        Optional image output path.
    """



    color_mapper = SinhalaCategoryColorMapper(
        classifier,
        similarity_threshold=0.80
    )



    wc = WordCloud(

        font_path=font_path,

        width=1200,

        height=700,

        background_color="white",

        max_words=300,

        relative_scaling=0.5,

        min_font_size=10,

        collocations=False

    )



    # Generate using frequency dictionary
    wc.generate_from_frequencies(
        frequencies
    )



    # ========================================================
    # Render Figure
    # ========================================================

    plt.figure(
        figsize=(14, 8)
    )


    plt.imshow(
        wc.recolor(
            color_func=color_mapper.color_func
        ),
        interpolation="bilinear"
    )


    plt.axis("off")



    # ========================================================
    # Custom Category Legend
    # ========================================================


    legend_items = []


    for category, color in CATEGORY_COLORS.items():

        legend_items.append(

            Patch(
                facecolor=color,
                label=category
            )

        )


    plt.legend(

        handles=legend_items,

        title="Transcript Categories",

        loc="lower center",

        bbox_to_anchor=(0.5, -0.08),

        ncol=2,

        frameon=False,

        fontsize=10

    )



    plt.tight_layout()



    if output_file:

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Word cloud saved: {output_file}"
        )


    plt.show()



# ============================================================
# Example Execution
# ============================================================


if __name__ == "__main__":


    # --------------------------------------------------------
    # Load Existing Classifier
    # --------------------------------------------------------

    classifier = SinhalaTelecomClassifier()



    # --------------------------------------------------------
    # Example cleaned Sinhala token frequencies
    # Normally generated from transcript preprocessing
    # --------------------------------------------------------

    sinhala_frequency = Counter({
        "බිල": 45,
        "ගෙවීම": 35,
        "සිග්නල්": 30,
        "නැහැ": 28,
        "රවුටර්": 22,
        "අලුත්": 18,
        "සේවාව": 15,
        "නම්බර්": 12
    })



    # --------------------------------------------------------
    # Sinhala Font
    # --------------------------------------------------------

    FONT_PATH = (
        r"D:\SLT- TEXT VISUALIZATION\fonts\static\NotoSansSinhala-Regular.ttf"
    )
    generate_sinhala_wordcloud(
        frequencies=sinhala_frequency,
        classifier=classifier,
        font_path=FONT_PATH,
        output_file="sinhala_category_wordcloud.png"
    )