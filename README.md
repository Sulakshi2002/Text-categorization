# Sinhala Transcript Word Cloud

This project creates Sinhala word cloud visualizations from transcript text files. It is useful for exploring the most frequent words in conversational data and highlighting important keywords.

## Features
- Reads transcript text files from the `test/` and `transcripts/` folders
- Removes common stop words to improve readability
- Supports Sinhala text rendering using a custom font
- Generates word cloud images for both general and keyword-focused analysis

## Project Files
- `sinhala_wordcloud.py` - Generates a standard word cloud from transcript text
- `sinhala_wordcloud_keywords.py` - Generates a keyword-focused word cloud
- `stop_words.txt` - Stop words used to filter out common terms
- `fonts/` - Sinhala font files used for rendering text
- `test/` and `transcripts/` - Sample transcript files used for testing and demonstration

## Requirements
Install the required Python packages:

```bash
pip install matplotlib wordcloud numpy pillow
```

## Usage
Run the main word cloud script:

```bash
python sinhala_wordcloud.py
```

This will generate `sinhala_wordcloud.png`.

Run the keyword-based version:

```bash
python sinhala_wordcloud_keywords.py
```

This will generate `sinhala_keyword_wordcloud.png`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
