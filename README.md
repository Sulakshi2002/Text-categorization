# Sinhala Transcript Word Cloud

This project generates a Sinhala word cloud from transcript text files to visualize the most frequent words in a collection of conversations.

## Features
- Reads transcript text files from a folder
- Removes common stop words
- Supports Sinhala text rendering with a custom font
- Produces a word cloud image

## Project Structure
- `sinhala_wordcloud.py` - Main script for generating the word cloud
- `stop_words.txt` - Stop words used to filter common terms
- `fonts/` - Font files for Sinhala text rendering
- `test/` and `transcripts/` - Sample transcript files

## Requirements
Install the required Python packages:

```bash
pip install matplotlib wordcloud numpy pillow
```

## Usage
Run the script from the project directory:

```bash
python sinhala_wordcloud.py
```

The generated word cloud image will be saved as `sinhala_wordcloud.png`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
