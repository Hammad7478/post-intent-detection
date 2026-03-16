import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def run_baseline(train_path, test_path, out_path):
    # Load data
    with open(train_path, 'r', encoding='utf-8') as f: 
        train = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f: 
        test = json.load(f)
        
    # Extract text and labels
    train_texts = [item['text'] for item in train]
    train_labels = [item['label'] for item in train]
    
    test_texts = [item['text'] for item in test]
    test_ids = [item['id'] for item in test]
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)
    
    # Make predictions
    preds = model.predict(X_test)
    
    # Format and save submission
    submission = [{"id": vid, "label": label} for vid, label in zip(test_ids, preds)]
    with open(out_path, 'w', encoding='utf-8') as f: 
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Logistic Regression baseline complete. Generated {len(submission)} predictions.")

if __name__ == "__main__":
    # Update paths as needed based on where you run this script
    run_baseline('../data/splits/train.json', '../data/splits/test.json', 'logistic_regression_submission.json')