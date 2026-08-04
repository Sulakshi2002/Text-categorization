# Sinhala Transcript Word Cloud

This project creates Sinhala word cloud visualizations from call-centre transcript text files. It is useful for exploring the most frequent words in conversational data, identifying keywords, and comparing different transcript categories.

## Features

- Reads transcript text files from the `transcripts/` folder
- Removes common stop words to improve readability
- Removes short tokens and transcript markers such as `[SI]`
- Supports Sinhala text rendering using a custom font or a Windows fallback font
- Generates a standard word cloud from cleaned transcript text
- Generates a category-focused word cloud using telecom classification patterns

## Project Files

- `preprocess.py` - Loads transcripts, cleans Sinhala text, removes stop words, and normalizes words
- `wordcloud_generator.py` - Generates a standard word cloud from all cleaned transcript tokens
- `wordcloud_analysis.py` - Generates a category-colored word cloud using the classifier dictionaries
- `classifier.py` - Provides telecom category matching for the analysis word cloud
- `stopwords/` - Sinhala and call-centre stopword lists
- `dictionaries/` - Telecom keyword dictionaries used for classification
- `fonts/` - Sinhala font files used for rendering text
- `transcripts/` - Sample transcript files used for testing and demonstration
- `outputs/` - Generated word cloud images

## Requirements

Install the required Python packages:

```bash
pip install matplotlib wordcloud numpy pillow
```

## Usage

Run the standard word cloud generator:

```bash
python wordcloud_generator.py
```

This will generate `outputs/sinhala_wordcloud.png`.

Run the category-based word cloud generator:

```bash
python wordcloud_analysis.py
```

This will generate `sinhala_category_wordcloud.png`.

## Notes

- The scripts resolve file paths from the script location, so they can be run from the `Text visualization/` folder directly.
- If `fonts/NotoSansSinhala-Regular.ttf` is not available, the generator falls back to installed Windows fonts such as `Nirmala.ttc` or `DejaVuSans.ttf`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
