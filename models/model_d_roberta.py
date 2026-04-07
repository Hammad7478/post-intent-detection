"""
model_d_roberta.py

Fine-tuning RoBERTa-base for high accuracy intent detection.
"""

import argparse
import json
import os
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

# Constants
MODEL_NAME = 'roberta-base' 
SEED = 42
MAX_LENGTH = 512

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.weights.to(logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train_mode(args):
    print(f"Loading data from {args.train}, {args.val}, {args.test}...")
    train_raw = load_data(args.train)
    val_raw = load_data(args.val)
    test_raw = load_data(args.test)

    le = LabelEncoder()
    all_labels_str = [item['label'] for item in train_raw]
    all_labels_encoded = le.fit_transform(all_labels_str)
    label_map = {label: int(i) for i, label in enumerate(le.classes_)}
    print(f"Label mapping: {label_map}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels_encoded),
        y=all_labels_encoded
    )
    print(f"Computed Class Weights: {class_weights}")

    def prepare_ds(raw):
        return Dataset.from_dict({
            'text': [item['text'] for item in raw],
            'label': le.transform([item['label'] for item in raw])
        })

    train_dataset = prepare_ds(train_raw)
    val_dataset = prepare_ds(val_raw)
    test_dataset = prepare_ds(test_raw)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LENGTH)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(le.classes_),
        id2label={i: label for i, label in enumerate(le.classes_)},
        label2id={label: i for i, label in enumerate(le.classes_)}
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        seed=SEED,
        logging_steps=10,
        push_to_hub=False,
        report_to="none",
        fp16=True # RoBERTa works fine with FP16
    )

    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print(f"\nStarting training (Model D: {MODEL_NAME})...")
    trainer.train()

    best_model_path = os.path.join(args.output_dir, 'best_model')
    trainer.save_model(best_model_path)
    with open(os.path.join(best_model_path, 'label_encoder.json'), 'w') as f:
        json.dump(label_map, f)
    print(f"\nModel and artifacts saved to {best_model_path}")

    print("\nEvaluating on TEST set...")
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids

    print(f"\n--- Model D ({MODEL_NAME}) on Test ---")
    print(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Macro F1: {f1_score(test_labels, test_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(test_labels, test_preds, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=le.classes_))

def predict_mode(args):
    model_path = args.model_path
    with open(os.path.join(model_path, 'label_encoder.json'), 'r') as f:
        label_map = json.load(f)
    id2label = {int(i): label for label, i in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ids = [item['id'] for item in data]
    texts = [item['text'] for item in data]

    results = []
    print(f"Predicting labels for {len(texts)} samples...")
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        for j, pred in enumerate(preds):
            results.append({"id": ids[i+j], "label": id2label[pred]})

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model D - RoBERTa Fine-tuning")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--train', default='data/data_splits/train.json')
    train_parser.add_argument('--val', default='data/data_splits/val.json')
    train_parser.add_argument('--test', default='data/data_splits/test.json')
    train_parser.add_argument('--output_dir', default='models/artifacts_d_roberta')
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--input', required=True)
    predict_parser.add_argument('--output', default='submission_d.json')
    predict_parser.add_argument('--model_path', default='models/artifacts_d_roberta/best_model')
    args = parser.parse_args()
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
