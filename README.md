# Smart-wallet-classifier

Live Demo: https://smart-wallet-classifier.streamlit.app/ 

Project Overview: A machine learning-powered web application that automatically categorizes financial transactions based on their descriptions using Python with scikit-learn, StreamLit for web framework and integrated TF-IDF vectorizer and Naive Bayes classifier.

# Dataset Summary
Mock transaction dataset generated using AI with 5 fields, `amount`, `merchant`, `timestamp`, `description`, `category`, with a total transactions of 1,722 records. The dataset includes 5 distinct spending categories:
- Food
- Transport
- Subscription
- Shopping 
- Utilities

# Model training steps and evaluation
## Data Preprocessing Approach
The preprocessing pipeline focuses entirely on text-based inputs, transaction descriptions contain the most important information:

- Text Cleaning: Convert to lowercase, remove punctuation, standardize whitespace
- Feature Extraction: TF-IDF vectorization with 300 most important terms
- Vocabulary Building: Include both single words and word pairs
- Stop Word Removal: Filter common English words

## Model Selection
Naive Bayes was selected as the final model. Naive Bayes was chosen for transaction classification because, it excels at text classification tasks due to its probabilistic approach to word patterns, making it ideal for analyzing transaction descriptions. Second, it performs efficiently with smaller datasets.

### Model Training Steps

1. **Data Loading and Exploration:** Loaded 1,722 transactions with 5 balanced categories, confirmed no missing values and consistent data quality.

2. **Text Preprocessing:** Cleaned transaction descriptions by converting to lowercase, removing punctuation, and standardizing whitespace.

3. **Feature Engineering:** Applied TF-IDF vectorization to convert text descriptions into numerical features, creating a vocabulary of 300 most important terms including both single words and word pairs.

4. **Data Splitting:** Divided dataset into 80% training (1,377 samples) and 20% testing (345 samples) with stratified sampling to maintain category balance.

5. **Model Training:** Trained Naive Bayes classifier on the processed features, achieving 99.4% accuracy on the test set.

6. **Model Validation:** Tested with real transaction examples to verify practical performance and saved the complete model pipeline for deployment.

## Performance Evaluation
The model demonstrates exceptional performance across all metrics and achieved an overall accuracy: 99.4%

# Setup and Usage Instruction
## Local Development
**Clone the repository**
git clone https://github.com/Pavitra2108/smart-wallet-classifier.git
cd smart-wallet-app

**Install dependencies**
pip install -r requirements.txt

**Run the application**
streamlit run smart-wallet-app/app.py

## Application Usage
- Enter transaction details in the form input
- Click "Classify Transaction" for category prediction
- View predicted category with confidence percentage
- Review transaction log with all classified entries
- Examine pie chart breakdown of spending by category

# Screenshot
  <img width="1810" height="858" alt="image" src="https://github.com/user-attachments/assets/313ce2f9-b023-41de-b07c-09985b110167" />

# Reflection
To scale for multiple users, the system would need a database to replace in-memory storage and an API architecture to handle concurrent requests. For bank integration, security becomes critical, requiring encryption, and compliance with financial regulations. The biggest technical challenge would be maintaining fast response times under high transaction volumes, likely requiring model caching and load balancing.
