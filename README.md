# Text correction & categorization
# Noisy Text Correction & Text Categorization (Sinhala NLP)

A lightweight Sinhala NLP project for **noisy text correction** and **intent classification**.
This system corrects misspelled or noisy Sinhala telecom-related text using fuzzy matching, then classifies the corrected conversation into predefined support categories such as **GB**, **Billing**, **Fault/Technical Assistance**, **Directory Queries**, and **Product Requests**.

## Features

* Sinhala noisy text correction using fuzzy matching (`difflib`)
* Dataset-based word validation
* Intent classification using keyword scoring
* Detects unmatched or unknown words
* Telecom customer support conversation categorization
* Simple and beginner-friendly Python implementation

## Categories Supported

* **GB** – Adding more Data GB
* **FAULT or TA** – Technical issues & fault reporting
* **DQ** – Directory and information queries
* **BILL** – Billing and payment related
* **PRODUCT** – Package and product related requests

## Technologies Used

* Python
* `difflib`
* String processing
* Rule-based NLP techniques

## Workflow

1. Load Sinhala custom word dataset
2. Correct noisy/misspelled Sinhala text
3. Detect unmatched words
4. Classify corrected text into categories
5. Display category scores and prediction

\

