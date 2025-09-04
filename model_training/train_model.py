# Simple Naive Bayes Training for Smart Wallet
# Just train one model - fast and effective!

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    """Load the processed data"""
    print("📂 Loading processed data...")
    
    try:
        X_train = pd.read_csv("model_data/X_train.csv")
        X_test = pd.read_csv("model_data/X_test.csv")
        y_train = pd.read_csv("model_data/y_train.csv")['y_train']
        y_test = pd.read_csv("model_data/y_test.csv")['y_test']
        categories = pd.read_csv("model_data/categories.csv")
        
        with open("model_data/tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        
        with open("model_data/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        print(f"✅ Training samples: {len(X_train)}")
        print(f"✅ Testing samples: {len(X_test)}")
        print(f"✅ Features: {X_train.shape[1]}")
        print(f"✅ Categories: {', '.join(categories['category'])}")
        
        return X_train, X_test, y_train, y_test, categories, tfidf, label_encoder
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def train_naive_bayes(X_train, X_test, y_train, y_test):
    """Train Naive Bayes model"""
    
    print(f"\n🤖 TRAINING NAIVE BAYES MODEL...")
    print("="*40)
    
    # Create and train model
    model = MultinomialNB()
    
    print("⏱️ Training...")
    model.fit(X_train, y_train)
    print("✅ Training complete!")
    
    # Make predictions
    print("🔮 Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Accuracy: {accuracy:.1%}")
    
    return model, y_pred, accuracy

def evaluate_model(model, y_test, y_pred, categories):
    """Evaluate the trained model"""
    
    print(f"\n📊 MODEL EVALUATION")
    print("="*40)
    
    category_names = categories['category'].tolist()
    
    # Classification report
    print("\n📈 Detailed Performance:")
    report = classification_report(y_test, y_pred, target_names=category_names)
    print(report)
    
    # Simple confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 Category-wise Accuracy:")
    
    for i, category in enumerate(category_names):
        correct = cm[i, i]
        total = cm[i, :].sum()
        accuracy = correct / total if total > 0 else 0
        print(f"   {category:<12}: {accuracy:.1%} ({correct}/{total} correct)")

def test_real_examples(model, tfidf, categories):
    """Test with real transaction examples"""
    
    print(f"\n🧪 TESTING WITH REAL EXAMPLES")
    print("="*40)
    
    test_examples = [
        "Coffee at Starbucks",
        "Uber ride downtown", 
        "Netflix subscription",
        "Grocery shopping",
        "Electric bill payment",
        "Gas station",
        "Amazon purchase",
        "Spotify monthly",
        "Restaurant dinner",
        "Bus pass"
    ]
    
    category_names = categories['category'].tolist()
    
    print("AI Predictions:")
    for example in test_examples:
        # Process text
        clean_text = example.lower()
        features = tfidf.transform([clean_text])
        
        # Make prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max()
        predicted_category = category_names[prediction]
        
        print(f"   '{example}' → {predicted_category} ({confidence:.1%})")

def save_model(model, tfidf, label_encoder, categories, accuracy):
    """Save the trained model"""
    
    print(f"\n💾 SAVING MODEL...")
    print("="*25)
    
    # Create folder
    if not os.path.exists("wallet_ai"):
        os.makedirs("wallet_ai")
        print("📁 Created wallet_ai/ folder")
    
    try:
        # Create AI package
        ai_package = {
            'model': model,
            'tfidf_vectorizer': tfidf,
            'label_encoder': label_encoder,
            'categories': categories['category'].tolist(),
            'accuracy': accuracy,
            'model_type': 'Naive Bayes',
            'created_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save AI system
        with open("wallet_ai/transaction_classifier.pkl", "wb") as f:
            pickle.dump(ai_package, f)
        
        # Create simple usage script
        usage_code = f'''# Smart Wallet Transaction Classifier
# Accuracy: {accuracy:.1%}

import pickle

# Load AI system
with open('wallet_ai/transaction_classifier.pkl', 'rb') as f:
    ai = pickle.load(f)

def classify_transaction(description):
    """Classify a transaction description"""
    # Clean text
    clean_text = description.lower().strip()
    
    # Convert to features
    features = ai['tfidf_vectorizer'].transform([clean_text])
    
    # Make prediction
    prediction = ai['model'].predict(features)[0]
    confidence = ai['model'].predict_proba(features)[0].max()
    
    # Get category name
    category = ai['categories'][prediction]
    
    return category, confidence

# Test examples
if __name__ == "__main__":
    test_transactions = [
        "Morning coffee at cafe",
        "Taxi ride to airport",
        "Monthly Netflix payment", 
        "Weekly groceries",
        "Electricity bill"
    ]
    
    print("🤖 Smart Wallet AI - Transaction Classification")
    print(f"📊 Model Accuracy: {['accuracy']:.1%}")
    print("-" * 50)
    
    for transaction in test_transactions:
        category, confidence = classify_transaction(transaction)
        print(f"'{"transaction"}' → {"category"} ({"confidence":.1%} confidence)")
'''
        
        with open("wallet_ai/use_classifier.py", "w") as f:
            f.write(usage_code)
        
        print("✅ transaction_classifier.pkl saved")
        print("✅ use_classifier.py saved (usage example)")
        
        # Show files created
        files = os.listdir("wallet_ai")
        print(f"\n📁 Files in wallet_ai/:")
        for file in files:
            size = os.path.getsize(f"wallet_ai/{file}")
            print(f"   📄 {file} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

def main():
    """Main training function"""
    
    print("🤖 SMART WALLET - NAIVE BAYES TRAINING")
    print("="*50)
    
    # Load data
    data = load_data()
    if data is None:
        print("❌ Please run simple_preprocessing.py first!")
        return
    
    X_train, X_test, y_train, y_test, categories, tfidf, label_encoder = data
    
    # Train model
    model, y_pred, accuracy = train_naive_bayes(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    evaluate_model(model, y_test, y_pred, categories)
    
    # Test with examples
    test_real_examples(model, tfidf, categories)
    
    # Save model
    save_success = save_model(model, tfidf, label_encoder, categories, accuracy)
    
    # Final summary
    print(f"\n" + "="*50)
    print("🎉 TRAINING COMPLETE!")
    print("="*50)
    print(f"🎯 Final Accuracy: {accuracy:.1%}")
    print(f"🤖 Model: Naive Bayes")
    
    if save_success:
        print(f"💾 AI saved: wallet_ai/transaction_classifier.pkl")
        print(f"📖 Usage: wallet_ai/use_classifier.py")
        print(f"\n🚀 YOUR TRANSACTION CLASSIFIER IS READY!")
    
    print(f"\n💡 Next: Build a simple web app!")
    
    return accuracy >= 0.8  # Return True if good accuracy

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Success! Your AI is ready for deployment.")
    else:
        print(f"\n⚠️ Consider improving the data or trying different features.")