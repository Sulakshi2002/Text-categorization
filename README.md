# Sinhala Telecom Voice-to-Text and Text Analysis

## About the Project

This project is developed to process Sinhala telecom customer calls in two main stages.

The first stage converts a customer's Sinhala voice recording into text using a locally stored **Wav2Vec2** speech recognition model.

The second stage takes the generated transcription and identifies what type of telecom inquiry the customer is making. For this, the system compares words in the transcription with predefined telecom keywords and calculates a similarity-based score.

The main idea is:

```text
Sinhala Voice
     ↓
Wav2Vec2
     ↓
Sinhala Transcription
     ↓
Text Analysis
     ↓
Inquiry Category
```

The system is designed to work locally, so an online speech-to-text service is not required during transcription.

---

## Project Parts

This project has two main parts:

### 1. Voice-to-Text Transcription

The voice-to-text component uses a trained **Wav2Vec2** model to convert Sinhala speech into text.

It accepts `.wav` audio files and produces the corresponding Sinhala transcription.

### 2. Text Analysis and Classification

The text analysis component takes the Sinhala transcription and checks which telecom category it is most closely related to.

The current categories are:

* Bill Inquiries
* Fault and Technical
* Product and New Service
* Telephone Number Request or Other

---

# 1. Sinhala Voice-to-Text

## How it works

The transcription system follows these steps:

```text
WAV Audio
   ↓
Load Audio
   ↓
Convert to 16 kHz
   ↓
Wav2Vec2 Processor
   ↓
Wav2Vec2 Model
   ↓
Decode Prediction
   ↓
Clean Transcription
   ↓
Sinhala Text
```

The audio is loaded using `librosa` and converted to a sampling rate of **16 kHz**, which is the sampling rate used by the model.

The Wav2Vec2 model then predicts the speech tokens, which are decoded into Sinhala text.

After decoding, the system removes `[UNK]` tokens and unnecessary spaces.

---

## Required Packages

Install the required Python libraries using:

```bash
pip install torch transformers librosa soundfile sentencepiece
pip install numpy
```

Or install everything from the project's `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Model Path

The Wav2Vec2 model is stored locally.

In the Python file, the model location is specified using:

```python
MODEL_PATH = r"D:\SLT\ASR\Wav2vec2 682 data\sinhala_asr_v1_final\sinhala_asr_v1_2026"
```

This path needs to be changed depending on where the model is stored on your computer.

For example:

```python
MODEL_PATH = r"C:\Models\sinhala_asr_v1_2026"
```

Make sure the selected folder contains the required Wav2Vec2 model and processor files.

---

## Supported Audio

Currently, the program expects:

```text
.wav
```

audio files.

The audio is automatically loaded at **16 kHz** before being passed to the model.

---

## Running the Transcription

Run the Python file:

```bash
python voice_to_text.py
```

The model will first be loaded:

```text
Loading model...
Model loaded successfully
```

The program will then ask for an audio file:

```text
Enter WAV path (or type 'q' to quit):
>
```

Enter the path of the WAV file:

```text
D:\Audio\call_01.wav
```

The transcription will then be displayed:

```text
====================
TRANSCRIPTION
====================

[Generated Sinhala transcription]
```

To stop the program, enter:

```text
q
```

---

# 2. Sinhala Telecom Text Analysis

The second part of the project analyzes the Sinhala transcription produced by the voice-to-text system.

Instead of using exact keyword matching only, the classifier also checks how similar a word is to the predefined keywords.

This is useful because speech recognition output can contain small spelling or transcription differences.

For example:

```text
Transcription word
       ↓
Compare with telecom keywords
       ↓
Calculate similarity
       ↓
Calculate category score
       ↓
Select the highest score
```

---

## Categories

The classifier currently uses four categories.

### Bill Inquiries

```text
Bill_Inquiries
```

Used for customer questions related to bills, payments, charges, and similar issues.

### Fault and Technical

```text
Fault_and_Technical
```

Used for technical problems and service faults.

### Product and New Service

```text
Product_And_New_Service
```

Used for inquiries about products, packages, data services, and new services.

### Telephone Number Request or Other

```text
Telephone_Number_Request_Or_Other
```

Used for telephone-number-related requests and other inquiries that do not clearly belong to the main categories.

---

# Category Files

The keywords used for classification are stored separately as JSON files.

The structure is:

```text
categories/
│
├── bill_inquiries.json
├── fault_and_technical.json
├── product_and_new_service.json
└── telephone_number_request_or_other.json
```

This makes it easier to update the keywords and weights without changing the main classifier code.

---

# How the Classifier Works

The classifier does not simply look for an exact word.

It performs the following steps:

```text
Input Transcription
        ↓
Normalize Text
        ↓
Split into Tokens
        ↓
Compare with Category Keywords
        ↓
Calculate Levenshtein Similarity
        ↓
Apply Keyword Weights
        ↓
Calculate Category Scores
        ↓
Select Highest Score
```

---

## Text Tokenization

The transcription is first normalized and divided into individual tokens.

The system supports:

* Sinhala words
* English words
* Numbers

The tokenizer uses Sinhala Unicode ranges together with English characters and numbers.

---

## Levenshtein Similarity

The project uses **Levenshtein distance** to deal with slightly different words.

Levenshtein distance measures how many changes are required to convert one word into another.

The system converts this distance into a similarity value between `0` and `1`.

For example:

```text
1.00  → Exact match
0.95  → Very similar
0.85  → Similar
0.50  → Not very similar
```

The current default matching threshold is:

```text
0.80
```

Therefore, a word needs to have at least `0.80` similarity with a keyword to be considered a match.

---

# Weighted Scoring

Each keyword in the category JSON files has an associated weight.

When a keyword matches a transcript word, the classifier calculates a score based on:

```text
Similarity
    +
Keyword Weight
    ↓
Category Score
```

The scores from matching words are added together for each category.

The category with the highest final score becomes the prediction.

---

# Example Output

When a transcription is entered, the system produces something similar to:

```text
Prediction Outcome   : BILL_INQUIRIES

Scoring Weights Matrix:
{
    "Bill_Inquiries": 15.40,
    "Fault_and_Technical": 3.20,
    "Product_And_New_Service": 0.00,
    "Telephone_Number_Request_Or_Other": 1.50
}

Matched Evidence Logs:
  - Token '...' matched with '...' in Bill_Inquiries
    (Sim: 0.91, Final Weight: 4.55)
```

The evidence section is useful because it shows **why the system selected a particular category**.

---

# Unknown Inquiries

Sometimes a transcription may not contain any words that match the available category keywords.

In that situation, the system does not force the transcription into one of the four categories.

Instead, it returns:

```text
UNKNOWN_DIRECT_INQUIRY
```

This allows unidentified or unsupported inquiries to be handled separately.

---

# Running the Text Classifier

Run:

```bash
python classifier.py
```

The program will display:

```text
======================================================================
       SINHALA TELECOM CLASSIFIER TERMINAL SYSTEM (2026)
   Type 'exit' or 'quit' to terminate the processing loop.
======================================================================
```

Enter the Sinhala transcription:

```text
Enter/Paste Sinhala Transcript Log:
>
```

The classifier will then display the predicted category, category scores, and matching evidence.

To stop the program, type:

```text
exit
```

or:

```text
quit
```

---

# Project Structure

A simple project structure can be maintained like this:

```text
Sinhala-Telecom-Analysis/
│
├── voice_to_text/
│   └── voice_to_text.py
│
├── text_analysis/
│   ├── classifier.py
│   │
│   └── categories/
│       ├── bill_inquiries.json
│       ├── fault_and_technical.json
│       ├── product_and_new_service.json
│       └── telephone_number_request_or_other.json
│
├── audio/
│   └── sample.wav
│
├── requirements.txt
└── README.md
```

The Wav2Vec2 model can be kept in a separate local directory rather than placing large model files directly inside the project repository.

---

# Requirements

The project uses Python and the following main libraries:

```text
Python
PyTorch
Hugging Face Transformers
Wav2Vec2
Librosa
SoundFile
NumPy
Regular Expressions
JSON
```

The main purpose of each library is:

| Library      | Use                                   |
| ------------ | ------------------------------------- |
| Python       | Main development language             |
| PyTorch      | Running the Wav2Vec2 model            |
| Transformers | Loading Wav2Vec2                      |
| Librosa      | Loading and processing audio          |
| SoundFile    | Audio file support                    |
| NumPy        | Numerical operations                  |
| JSON         | Storing category keywords and weights |
| `re`         | Text processing and tokenization      |

---

# Complete Workflow

The complete system can be viewed as:

```text
                  Customer Call
                       │
                       ▼
                Sinhala Voice
                       │
                       ▼
                WAV Audio File
                       │
                       ▼
              ┌─────────────────┐
              │    Wav2Vec2     │
              │  Voice-to-Text  │
              └─────────────────┘
                       │
                       ▼
              Sinhala Transcription
                       │
                       ▼
              ┌─────────────────┐
              │  Text Analysis  │
              └─────────────────┘
                       │
                       ▼
              Keyword Matching
                       │
                       ▼
             Levenshtein Similarity
                       │
                       ▼
               Weighted Scoring
                       │
                       ▼
              Inquiry Category
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           Bill     Technical   Product
                       │
                       ▼
                 Other / Unknown
```

---

# Current Status

## Voice-to-Text

The following parts are currently implemented:

* Local Wav2Vec2 model loading
* WAV audio input
* 16 kHz audio processing
* Sinhala speech recognition
* Text decoding
* `[UNK]` removal
* Duplicate-space cleanup
* Multiple audio processing through the terminal

## Text Analysis

The following parts are currently implemented:

* Sinhala text tokenization
* Category-based keyword matching
* Levenshtein similarity
* Similarity threshold
* Keyword weighting
* Category score calculation
* Evidence logging
* Unknown inquiry handling
* JSON-based category configuration

---

# Future Improvements

Some possible improvements for the next stage of the project are:

* Process a large number of audio files automatically.
* Save generated transcriptions to CSV or JSON.
* Connect the voice-to-text output directly to the text classifier.
* Improve Sinhala text preprocessing.
* Add more telecom keywords and categories.
* Tune the similarity threshold using a larger dataset.
* Add confidence scores to predictions.
* Compare the current approach with machine-learning classification models.
* Generate charts showing the most common customer inquiry types.
* Build a dashboard for analyzing call-center transcriptions.
* Add support for more audio formats.

---

# Notes

The voice-to-text component uses a trained local Wav2Vec2 model.

The current text classification component is **not a trained ML model**. It is a **rule-based classification approach using telecom keywords, Levenshtein similarity, and weighted scoring**.

This approach was selected because it is simple, interpretable, and easy to update as new telecom inquiry patterns are identified.

---

# Project Purpose

The overall purpose of this project is to make Sinhala telecom call data easier to process and analyze.

Instead of manually listening to every call:

```text
Voice Call
   ↓
Automatic Transcription
   ↓
Automatic Text Analysis
   ↓
Inquiry Category
```

This can help reduce manual work and make large collections of Sinhala telecom call transcriptions easier to analyze.
