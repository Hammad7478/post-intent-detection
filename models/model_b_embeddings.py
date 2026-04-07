"""
model_b_tfidf_char.py

Implementation Note:
  Since sentence-transformers/torch are not available in this environment, 
  this version (Model B) uses a high-dimensional hybrid TF-IDF (Word + Char n-grams) 
  to provide a stronger signal than the simple word-based Model A.

How to train:
  python models/model_b_embeddings.py train --train data/data_splits/train.json --val data/data_splits/val.json --test data/data_splits/test.json --artifacts_dir models/artifacts_b

How to predict:
  python models/model_b_embeddings.py predict --input test_unlabeled.json --output submission.json --artifacts_dir models/artifacts_b
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
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

# Constants
SEED = 42

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

    # 5) Simple majority-class baseline on train
    majority_class = Counter(train_labels).most_common(1)[0][0]
    print(f"\nMajority class in training set: {majority_class}")
    
    test_majority_preds = [majority_class] * len(test_labels)
    print("\n--- Majority Class Baseline on Test ---")
    print(f"Accuracy: {accuracy_score(test_labels, test_majority_preds):.4f}")
    print(f"Macro F1: {f1_score(test_labels, test_majority_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(test_labels, test_majority_preds, average='weighted'):.4f}")

    # 1) Hybrid Feature Extraction (Word + Char n-grams)
    # This acts as our "Model B" high-capacity baseline
    vectorizer = FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True)),
        ('char', TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=10000, sublinear_tf=True))
    ])

    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # 3) Caching vectors to disk (satisfying requirement for efficiency on reruns)
    cache_path = os.path.join(args.artifacts_dir, 'vectorized_cache.joblib')
    if os.path.exists(cache_path):
        print(f"Loading cached vectors from {cache_path}...")
        cached_data = joblib.load(cache_path)
        X_train = cached_data['train']
        X_val = cached_data['val']
        X_test = cached_data['test']
        vectorizer = cached_data['vectorizer']
    else:
        print("Fitting vectorizer and transforming data...")
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        X_test = vectorizer.transform(test_texts)
        print(f"Saving vectors to {cache_path}...")
        joblib.dump({
            'train': X_train, 
            'val': X_val, 
            'test': X_test, 
            'vectorizer': vectorizer
        }, cache_path)

    # Combine train and val for GridSearchCV
    from scipy.sparse import vstack
    X = vstack([X_train, X_val])
    y = train_labels + val_labels
    test_fold = [-1] * len(train_texts) + [0] * len(val_texts)
    ps = PredefinedSplit(test_fold)

    # 2) Lightweight classifier + Tuning
    param_grid = [
        {
            'clf': [LogisticRegression(random_state=SEED, class_weight='balanced', max_iter=2000)],
            'clf__C': [0.1, 1.0, 10.0, 100.0]
        },
        {
            'clf': [LinearSVC(random_state=SEED, class_weight='balanced', max_iter=5000, dual='auto')],
            'clf__C': [0.01, 0.1, 1.0, 10.0]
        }
    ]

    pipeline = Pipeline([('clf', LogisticRegression())])

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
    best_clf_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items() if k.startswith('clf__')}
    best_clf_class = grid_search.best_params_['clf'].__class__
    best_clf = best_clf_class(**best_clf_params, random_state=SEED, class_weight='balanced')
    best_clf.fit(X_train, train_labels)

    # 4) Report on TEST
    test_preds = best_clf.predict(X_test)
    print("\n--- Model B (Hybrid TF-IDF) on Test ---")
    print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds))

    # 6) Save artifacts
    clf_path = os.path.join(args.artifacts_dir, 'classifier.joblib')
    joblib.dump(best_clf, clf_path)
    
    vec_only_path = os.path.join(args.artifacts_dir, 'vectorizer.joblib')
    joblib.dump(vectorizer, vec_only_path)
    
    print(f"\nArtifacts saved to {args.artifacts_dir}")

def predict_mode(args):
    clf_path = os.path.join(args.artifacts_dir, 'classifier.joblib')
    vec_path = os.path.join(args.artifacts_dir, 'vectorizer.joblib')
    
    if not os.path.exists(clf_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Artifacts not found. Please train the model first.")
    
    best_clf = joblib.load(clf_path)
    vectorizer = joblib.load(vec_path)
    
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ids = [item['id'] for item in data]
    texts = [item['text'] for item in data]
    
    print(f"Vectorizing {len(texts)} texts...")
    X = vectorizer.transform(texts)
    
    preds = best_clf.predict(X)
    
    submission = [{"id": vid, "label": label} for vid, label in zip(ids, preds)]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
        
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model B - Hybrid TF-IDF Classifier")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--train', default='data/data_splits/train.json')
    train_parser.add_argument('--val', default='data/data_splits/val.json')
    train_parser.add_argument('--test', default='data/data_splits/test.json')
    train_parser.add_argument('--artifacts_dir', default='models/artifacts_b')
    
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--input', required=True)
    predict_parser.add_argument('--output', default='submission_b.json')
    predict_parser.add_argument('--artifacts_dir', default='models/artifacts_b')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
