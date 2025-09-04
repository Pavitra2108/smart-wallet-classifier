# SIMPLE Smart Wallet Data Preprocessing
# Just the essentials: Load → Process → Save

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def main():
    print("🚀 SIMPLE SMART WALLET PREPROCESSING")
    print("="*50)
    
    # STEP 1: Load Data
    print("📂 Loading data...")
    try:
        df = pd.read_csv("Transaction_Dataset.csv")
        print(f"✅ Loaded {len(df)} transactions")
        print(f"✅ Categories: {list(df['Category'].unique())}")
    except:
        print("❌ Error: Make sure 'Transaction_Dataset.csv' is in the same folder!")
        return
    
    # STEP 2: Process Text
    print("\n🔧 Processing descriptions...")
    
    # Clean text descriptions
    df['description_clean'] = df['Description'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    X = tfidf.fit_transform(df['description_clean'])
    X_df = pd.DataFrame(X.toarray(), columns=[f'word_{i}' for i in range(X.shape[1])])
    
    # Encode categories
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])
    
    print(f"✅ Created {X.shape[1]} text features")
    print(f"✅ Encoded {len(le.classes_)} categories")
    
    # STEP 3: Split Data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    print(f"✅ Training: {len(X_train)} samples")
    print(f"✅ Testing: {len(X_test)} samples")
    
    # STEP 4: Save Files (ESSENTIAL!)
    print("\n💾 SAVING FILES...")
    
    # Create folder
    if not os.path.exists("model_data"):
        os.makedirs("model_data")
        print("📁 Created model_data/ folder")
    
    try:
        # Save the most important files
        X_train.to_csv("model_data/X_train.csv", index=False)
        X_test.to_csv("model_data/X_test.csv", index=False)
        pd.DataFrame({'y_train': y_train}).to_csv("model_data/y_train.csv", index=False)
        pd.DataFrame({'y_test': y_test}).to_csv("model_data/y_test.csv", index=False)
        
        # Save model objects
        with open("model_data/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        
        with open("model_data/label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        
        # Save category mapping
        categories = pd.DataFrame({
            'number': range(len(le.classes_)),
            'category': le.classes_
        })
        categories.to_csv("model_data/categories.csv", index=False)
        
        print("✅ X_train.csv saved")
        print("✅ X_test.csv saved") 
        print("✅ y_train.csv saved")
        print("✅ y_test.csv saved")
        print("✅ tfidf_vectorizer.pkl saved")
        print("✅ label_encoder.pkl saved")
        print("✅ categories.csv saved")
        
        # Verify files exist
        files = os.listdir("model_data")
        print(f"\n🎯 SUCCESS! Created {len(files)} files in model_data/")
        
        # Quick test
        print("\n🧪 QUICK TEST:")
        test_desc = "coffee at starbucks"
        test_features = tfidf.transform([test_desc.lower()])
        print(f"✅ Text processing works: '{test_desc}' → {test_features.shape[1]} features")
        
        print("\n" + "="*50)
        print("🎉 PREPROCESSING COMPLETE!")
        print("="*50)
        print("📁 Files saved in: model_data/")
        print("🚀 Ready for model training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n💡 Next: Run model training code!")
    else:
        print("\n❌ Fix the errors above before proceeding.")