# Smart-wallet-classifier

Live Demo: https://smart-wallet-classifier.streamlit.app/

Project Overview: A machine learning-powered web application that automatically categorizes financial transactions based on their descriptions.

# Dataset Summary
Mock transaction dataset generated using AI with 5 fields, `amount`, `merchant`, `timestamp`, `description`, `category`
Total Transactions: 1,722 records
Categories: 5 distinct spending categories
- Food
- Transport
- Subscription
- Shopping 
- Utilities

# Model training steps and evaluation
## Data Preprocessing Approach
The preprocessing pipeline focuses entirely on text-based inputs, transaction descriptions contain the most important information:

Text Cleaning: Convert to lowercase, remove punctuation, standardize whitespace
Feature Extraction: TF-IDF vectorization with 300 most important terms
Vocabulary Building: Include both single words and word pairs
Stop Word Removal: Filter common English words

## Model Selection
Naive Bayes was selected as the final model

