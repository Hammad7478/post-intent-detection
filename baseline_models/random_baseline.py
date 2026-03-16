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
    # Update paths as needed based on where you run this script
    run_baseline('../data/splits/train.json', '../data/splits/test.json', 'random_submission.json')