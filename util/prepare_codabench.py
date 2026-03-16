import json
import os
import zipfile

def create_unlabeled(in_path, out_path):
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    unlabeled = []
    for item in data:
        unlabeled.append({"id": item["id"], "text": item["text"]})
        
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(unlabeled, f, indent=2, ensure_ascii=False)

def create_codabench_bundle():
    base_dir = "codabench_bundle"
    os.makedirs(f"{base_dir}/data", exist_ok=True)
    os.makedirs(f"{base_dir}/reference", exist_ok=True)
    os.makedirs(f"{base_dir}/scoring_program", exist_ok=True)
    os.makedirs(f"{base_dir}/starting_kit", exist_ok=True)

    # 1. Data
    with open("data/splits/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(f"{base_dir}/data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    create_unlabeled("data/splits/val.json", f"{base_dir}/data/val_unlabeled.json")
    create_unlabeled("data/splits/test.json", f"{base_dir}/data/test_unlabeled.json")

    # 2. Reference
    with open("data/splits/val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(f"{base_dir}/reference/val_reference.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    # 3. Scoring Program
    scoring_script = """
import json
import sys
import os

def calculate_metrics(y_true, y_pred):
    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true) if y_true else 0
    
    # Simple F1 Weighted implementation
    labels = list(set(y_true))
    f1_sum = 0
    total_samples = len(y_true)
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        weight = sum(1 for t in y_true if t == label) / total_samples
        f1_sum += f1 * weight
        
    return acc, f1_sum

def evaluate(ref_dir, sub_dir, output_dir):
    print(f"Scoring started. Looking in {ref_dir} and {sub_dir}")
    
    ref_file = os.path.join(ref_dir, 'val_reference.json')
    sub_file = os.path.join(sub_dir, 'submission.json')
    
    if not os.path.exists(ref_file):
        print(f"ERROR: Reference file not found at {ref_file}")
        # List files for debugging
        if os.path.exists(ref_dir):
            print(f"Files in ref_dir: {os.listdir(ref_dir)}")
        return

    if not os.path.exists(sub_file):
        print(f"ERROR: Submission file not found at {sub_file}")
        if os.path.exists(sub_dir):
            print(f"Files in sub_dir: {os.listdir(sub_dir)}")
        return

    print("Loading data...")
    with open(ref_file, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
        ref = {item['id']: item['label'] for item in ref_data}
    
    with open(sub_file, 'r', encoding='utf-8') as f:
        sub_data = json.load(f)
        sub = {item['id']: item['label'] for item in sub_data}
    
    print(f"Processing {len(ref)} reference items and {len(sub)} submission items.")
    
    ids = list(ref.keys())
    y_true = [ref[i] for i in ids]
    y_pred = [sub.get(i, 'OTHER') for i in ids]
    
    acc, f1 = calculate_metrics(y_true, y_pred)
    
    # Convert to percentages
    acc_pct = acc * 100
    f1_pct = f1 * 100
    
    print(f"Scores calculated - Accuracy: {acc_pct:.2f}%, F1: {f1_pct:.2f}%")
    
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as f:
        f.write(f"accuracy: {acc_pct}\\n")
        f.write(f"f1_weighted: {f1_pct}\\n")
    print("Scores written to scores.txt")

if __name__ == "__main__":
    # Codabench typically passes 1: input_dir, 2: output_dir
    # But if they are missing, we use the standard default paths
    try:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    except IndexError:
        print("WARNING: Arguments missing. Falling back to default Codabench paths.")
        input_dir = '/app/input'
        output_dir = '/app/output'
    
    # In some environments, ref and res are directly in the input_dir
    # In others, they are in subdirectories. We will check both.
    ref_path = os.path.join(input_dir, 'ref')
    res_path = os.path.join(input_dir, 'res')
    
    # Fallback: if 'ref' doesn't exist inside input_dir, maybe input_dir IS the ref dir
    if not os.path.exists(ref_path):
        ref_path = input_dir
    if not os.path.exists(res_path):
        res_path = input_dir
        
    evaluate(ref_path, res_path, output_dir)
"""
    with open(f"{base_dir}/scoring_program/score.py", "w") as f:
        f.write(scoring_script.strip())
    with open(f"{base_dir}/scoring_program/metadata.yaml", "w") as f:
        f.write("command: python3 score.py\n")

    # 4. Starting Kit (Random Baseline)
    starting_kit_code = """
import json
import random

def run_baseline(train_path, test_path, out_path):
    # Load labels from training data to know what's available
    with open(train_path, 'r', encoding='utf-8') as f: 
        train = json.load(f)
    labels = list(set(item['label'] for item in train))
    
    # Load unlabeled data
    with open(test_path, 'r', encoding='utf-8') as f: 
        test = json.load(f)
    
    # Randomly assign a label to each ID
    submission = []
    for item in test:
        submission.append({
            "id": item["id"],
            "label": random.choice(labels)
        })
        
    with open(out_path, 'w', encoding='utf-8') as f: 
        json.dump(submission, f, indent=2, ensure_ascii=False)
    
    print(f"Random baseline complete. Generated {len(submission)} predictions.")

if __name__ == "__main__":
    # In the starting kit, these files are in the same folder
    run_baseline('train.json', 'val_unlabeled.json', 'submission.json')
"""
    with open(f"{base_dir}/starting_kit/baseline.py", "w") as f:
        f.write(starting_kit_code.strip())
    with open(f"{base_dir}/starting_kit/metadata.yaml", "w") as f:
        f.write("command: python3 baseline.py\n")

    # 5. Competition YAML (Maximal Schema)
    yaml_content = """
version: 2
title: "Post Intent Detection (Group 27)"
description: "Classify the communicative intent of Reddit posts."
image: logo.jpg
terms: terms.md
starting_kit: starting_kit.zip
make_data_available: True
make_programs_available: True
has_registration: True
pages:
  - title: Overview
    file: overview.md
  - title: Data
    file: data.md
  - title: Evaluation
    file: evaluation.md
tasks:
  - index: 0
    name: "Intent Detection Task"
    description: "Predict Reddit intent."
    data: data.zip
    reference_data: reference.zip
    scoring_program: scoring_program.zip
    starting_kit: starting_kit.zip
phases:
  - index: 0
    name: "Validation"
    description: "Validation Phase"
    start: "2026-03-14T00:00:00+00:00"
    starting_kit: starting_kit.zip
    public_data: data.zip
    tasks: [0]
leaderboards:
  - index: 0
    title: Results
    key: main
    columns:
      - title: Accuracy (%)
        key: accuracy
        index: 0
      - title: F1 (Weighted %)
        key: f1_weighted
        index: 1"""
    with open(f"{base_dir}/competition.yaml", "w") as f:
        f.write(yaml_content.strip())

    # 6. Documentation
    with open(f"{base_dir}/overview.md", "w") as f:
        f.write("# Overview\n\nTo get started, please go to the **Participate** tab and download the **Starting Kit**.")
    with open(f"{base_dir}/data.md", "w") as f:
        f.write("# Data\nDataset contains Reddit posts.")
    with open(f"{base_dir}/evaluation.md", "w") as f:
        f.write("# Evaluation\nScored on Accuracy.")
    with open(f"{base_dir}/terms.md", "w") as f:
        f.write("# Terms\nCOMPSCI 4NL3 Winter 2026.")

    print("Codabench bundle structure updated.")

if __name__ == "__main__":
    create_codabench_bundle()
