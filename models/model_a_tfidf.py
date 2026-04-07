"""
model_a_tfidf.py

How to train:
  python models/model_a_tfidf.py train --train data/data_splits/train.json --val data/data_splits/val.json --test data/data_splits/test.json --artifacts_dir models/artifacts

How to predict:
  python models/model_a_tfidf.py predict --input test_unlabeled.json --output submission.json --artifacts_dir models/artifacts
"""

import argparse
import json
import os
import joblib
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, GridSearchCV

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [item.get('label', None) for item in data]
    ids = [item['id'] for item in data]
    return ids, texts, labels

def train_mode(args):
    print("Loading data...")
    train_ids, train_texts, train_labels = load_data(args.train)
    val_ids, val_texts, val_labels = load_data(args.val)
    test_ids, test_texts, test_labels = load_data(args.test)

    # 4) Simple majority-class baseline on train
    majority_class = Counter(train_labels).most_common(1)[0][0]
    print(f"\nMajority class in training set: {majority_class}")
    
    test_majority_preds = [majority_class] * len(test_labels)
    print("\n--- Majority Class Baseline on Test ---")
    print(f"Accuracy: {accuracy_score(test_labels, test_majority_preds):.4f}")
    print(f"Macro F1: {f1_score(test_labels, test_majority_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(test_labels, test_majority_preds, average='weighted'):.4f}")
    
    # Combine train and val for GridSearchCV using PredefinedSplit
    X = train_texts + val_texts
    y = train_labels + val_labels
    
    # -1 for train, 0 for val
    test_fold = [-1] * len(train_texts) + [0] * len(val_texts)
    ps = PredefinedSplit(test_fold)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])
    
    param_grid = [
        {
            'tfidf__ngram_range': [(1, 2), (1, 3)],
            'tfidf__min_df': [1, 2],
            'tfidf__sublinear_tf': [True, False],
            'clf': [LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)],
            'clf__C': [0.1, 1.0, 10.0]
        },
        {
            'tfidf__ngram_range': [(1, 2), (1, 3)],
            'tfidf__min_df': [1, 2],
            'tfidf__sublinear_tf': [True, False],
            'clf': [LinearSVC(random_state=42, class_weight='balanced', max_iter=2000)],
            'clf__C': [0.1, 1.0, 10.0]
        }
    ]
    
    print("\nStarting hyperparameter tuning on Validation set...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=ps, 
        scoring='f1_macro', 
        n_jobs=-1, 
        verbose=1, 
        refit=False
    )
    grid_search.fit(X, y)
    
    print(f"\nBest parameters found:\n{grid_search.best_params_}")
    print(f"Best validation Macro F1: {grid_search.best_score_:.4f}")
    
    print("\nRetraining best model on TRAIN set only...")
    best_model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression()) # Dummy placeholder, will be overwritten
    ])
    best_model.set_params(**grid_search.best_params_)
    best_model.fit(train_texts, train_labels)
    
    # 3) Report on TEST once
    test_preds = best_model.predict(test_texts)
    print("\n--- Best Model on Test ---")
    print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds))
    
    # 5) Save artifacts
    os.makedirs(args.artifacts_dir, exist_ok=True)
    model_path = os.path.join(args.artifacts_dir, 'model_a.joblib')
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")

def predict_mode(args):
    model_path = os.path.join(args.artifacts_dir, 'model_a.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    best_model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ids = [item['id'] for item in data]
    texts = [item['text'] for item in data]
    
    preds = best_model.predict(texts)
    
    submission = [{"id": vid, "label": label} for vid, label in zip(ids, preds)]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
        
    print(f"Predictions saved to {args.output} ({len(preds)} samples)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model A - TF-IDF Text Classifier")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    train_parser = subparsers.add_parser('train', help="Train the model and save artifacts")
    train_parser.add_argument('--train', default='data/data_splits/train.json', help="Path to training data")
    train_parser.add_argument('--val', default='data/data_splits/val.json', help="Path to validation data")
    train_parser.add_argument('--test', default='data/data_splits/test.json', help="Path to test data")
    train_parser.add_argument('--artifacts_dir', default='models/artifacts', help="Directory to save model artifacts")
    
    predict_parser = subparsers.add_parser('predict', help="Predict labels for unlabeled data")
    predict_parser.add_argument('--input', required=True, help="Path to unlabeled input JSON")
    predict_parser.add_argument('--output', default='submission.json', help="Path to save submission JSON")
    predict_parser.add_argument('--artifacts_dir', default='models/artifacts', help="Directory to load model artifacts")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
