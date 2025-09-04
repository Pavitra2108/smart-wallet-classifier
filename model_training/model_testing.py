# Manual Model Testing - Interactive Testing Script
# Test your Smart Wallet AI with real transaction descriptions

import pickle
import pandas as pd
import os
from datetime import datetime

def load_ai_model():
    """Load the trained AI model"""
    
    try:
        # Try loading from wallet_ai folder first
        if os.path.exists("wallet_ai/transaction_classifier.pkl"):
            with open("wallet_ai/transaction_classifier.pkl", "rb") as f:
                ai = pickle.load(f)
            print("✅ AI model loaded successfully!")
            return ai
        else:
            print("❌ Could not find the AI model file.")
            print("   Please make sure you ran the training script first.")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def classify_transaction(ai, description):
    """Classify a transaction description"""
    
    try:
        # Clean text
        clean_text = description.lower().strip()
        
        # Convert to features
        features = ai['tfidf_vectorizer'].transform([clean_text])
        
        # Make prediction
        prediction = ai['model'].predict(features)[0]
        probabilities = ai['model'].predict_proba(features)[0]
        confidence = probabilities.max()
        
        # Get category name
        category = ai['categories'][prediction]
        
        # Get top 3 predictions for more insight
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_predictions = []
        for idx in top_3_indices:
            cat_name = ai['categories'][idx]
            conf = probabilities[idx]
            top_3_predictions.append((cat_name, conf))
        
        return category, confidence, top_3_predictions
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, 0, []

def display_prediction_results(description, prediction, confidence, top_3):
    """Display prediction results in a nice format"""
    
    print("\n" + "="*60)
    print(f"📝 Transaction: '{description}'")
    print("-"*60)
    print(f"🎯 Prediction: {prediction} ({confidence:.1%} confidence)")
    
    if len(top_3) > 1:
        print(f"\n📊 Top 3 possibilities:")
        for i, (category, conf) in enumerate(top_3, 1):
            confidence_bar = "█" * int(conf * 20)  # Visual confidence bar
            print(f"   {i}. {category:<12}: {conf:.1%} {confidence_bar}")
    
    print("="*60)

def get_user_feedback():
    """Get feedback from user about the prediction"""
    
    while True:
        print("\n🤔 Is this prediction correct?")
        print("   1. ✅ Correct")
        print("   2. ❌ Wrong")
        print("   3. 🤷 Not sure")
        print("   4. ℹ️  Show categories list")
        
        choice = input("\nYour choice (1-4): ").strip()
        
        if choice == '1':
            return 'correct', None
        elif choice == '2':
            return 'wrong', get_correct_category()
        elif choice == '3':
            return 'unsure', None
        elif choice == '4':
            show_categories()
            continue
        else:
            print("⚠️ Please enter 1, 2, 3, or 4")
            continue

def show_categories():
    """Show available categories"""
    categories = ['Food', 'Transport', 'Subscription', 'Shopping', 'Utilities']
    print("\n📁 Available categories:")
    for i, cat in enumerate(categories, 1):
        print(f"   {i}. {cat}")

def get_correct_category():
    """Get the correct category from user"""
    categories = ['Food', 'Transport', 'Subscription', 'Shopping', 'Utilities']
    
    while True:
        show_categories()
        try:
            choice = input("\nWhat's the correct category? (1-5): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= 5:
                return categories[choice_num - 1]
            else:
                print("⚠️ Please enter a number between 1-5")
        except ValueError:
            print("⚠️ Please enter a valid number")

def save_test_results(test_results):
    """Save test results for analysis"""
    
    if not test_results:
        print("No test results to save.")
        return
    
    try:
        # Create results folder
        if not os.path.exists("test_results"):
            os.makedirs("test_results")
        
        # Convert to DataFrame
        df = pd.DataFrame(test_results)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/manual_testing_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Show summary
        total_tests = len(test_results)
        correct_tests = len([r for r in test_results if r['feedback'] == 'correct'])
        wrong_tests = len([r for r in test_results if r['feedback'] == 'wrong'])
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Total tests: {total_tests}")
        print(f"   Correct predictions: {correct_tests} ({correct_tests/total_tests:.1%})")
        print(f"   Wrong predictions: {wrong_tests} ({wrong_tests/total_tests:.1%})")
        print(f"   Results saved: {filename}")
        
        # Show common mistakes
        if wrong_tests > 0:
            print(f"\n❌ COMMON MISTAKES:")
            wrong_results = [r for r in test_results if r['feedback'] == 'wrong']
            for result in wrong_results:
                print(f"   '{result['description']}' → Predicted: {result['predicted']}, Actual: {result['actual']}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def interactive_testing_session(ai):
    """Run an interactive testing session"""
    
    print("\n🧪 INTERACTIVE TESTING SESSION")
    print("="*60)
    print("Type transaction descriptions to test your AI!")
    print("Examples: 'coffee at starbucks', 'uber to airport', 'netflix payment'")
    print("Type 'quit' to end the session")
    print("="*60)
    
    test_results = []
    
    while True:
        # Get transaction description
        print(f"\n📝 Test #{len(test_results) + 1}")
        description = input("\nEnter transaction description (or 'quit'): ").strip()
        
        if description.lower() in ['quit', 'exit', 'q']:
            break
        
        if not description:
            print("⚠️ Please enter a transaction description")
            continue
        
        # Make prediction
        prediction, confidence, top_3 = classify_transaction(ai, description)
        
        if prediction is None:
            print("❌ Could not make prediction")
            continue
        
        # Display results
        display_prediction_results(description, prediction, confidence, top_3)
        
        # Get user feedback
        feedback, actual_category = get_user_feedback()
        
        # Store results
        test_result = {
            'description': description,
            'predicted': prediction,
            'confidence': confidence,
            'feedback': feedback,
            'actual': actual_category if actual_category else prediction,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        test_results.append(test_result)
        
        print(f"✅ Feedback recorded: {feedback}")
    
    return test_results

def quick_batch_test():
    """Quick test with predefined examples"""
    
    test_examples = [
        "Morning coffee at local cafe",
        "Taxi ride to the airport",
        "Netflix monthly subscription fee", 
        "Weekly grocery shopping",
        "Internet bill payment",
        "Gas station refuel",
        "Amazon book purchase",
        "Spotify premium monthly",
        "Lunch at McDonald's",
        "Bus pass for commuting"
    ]
    
    print("\n⚡ QUICK BATCH TEST")
    print("="*50)
    print("Testing with common transaction examples...")
    
    return test_examples

def main():
    """Main testing function"""
    
    print("🧪 SMART WALLET AI - MANUAL TESTING")
    print("="*60)
    
    # Load AI model
    ai = load_ai_model()
    if ai is None:
        return
    
    print(f"📊 Model info:")
    print(f"   Accuracy: {ai.get('accuracy', 'Unknown'):.1%}")
    print(f"   Categories: {', '.join(ai['categories'])}")
    print(f"   Created: {ai.get('created_date', 'Unknown')}")
    
    while True:
        print(f"\n🎯 TESTING OPTIONS:")
        print("   1. Interactive testing (type your own examples)")
        print("   2. Quick batch test (predefined examples)")
        print("   3. Single test")
        print("   4. Quit")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == '1':
            test_results = interactive_testing_session(ai)
            save_test_results(test_results)
            
        elif choice == '2':
            examples = quick_batch_test()
            test_results = []
            
            for description in examples:
                prediction, confidence, top_3 = classify_transaction(ai, description)
                display_prediction_results(description, prediction, confidence, top_3)
                feedback, actual = get_user_feedback()
                
                test_results.append({
                    'description': description,
                    'predicted': prediction,
                    'confidence': confidence,
                    'feedback': feedback,
                    'actual': actual if actual else prediction,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            save_test_results(test_results)
            
        elif choice == '3':
            description = input("\nEnter transaction description: ").strip()
            if description:
                prediction, confidence, top_3 = classify_transaction(ai, description)
                display_prediction_results(description, prediction, confidence, top_3)
                feedback, actual = get_user_feedback()
                print(f"✅ Feedback recorded: {feedback}")
            
        elif choice == '4':
            print("👋 Thanks for testing! Use the results to improve your AI.")
            break
            
        else:
            print("⚠️ Please choose 1, 2, 3, or 4")

if __name__ == "__main__":
    main()