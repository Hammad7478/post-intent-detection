import json
from collections import Counter

def run_baseline(train_path, test_path, out_path):
    # Load training data to find the majority class
    with open(train_path, 'r', encoding='utf-8') as f: 
        train = json.load(f)
    
    # Count label frequencies and find the most common one
    label_counts = Counter(item['label'] for item in train)
    majority_label = label_counts.most_common(1)[0][0]
    
    # Load unlabeled test data
    with open(test_path, 'r', encoding='utf-8') as f: 
        test = json.load(f)
    
    # Assign the majority label to every instance
    submission = []
    for item in test:
        submission.append({
            "id": item["id"],
            "label": majority_label
        })
        
    with open(out_path, 'w', encoding='utf-8') as f: 
        json.dump(submission, f, indent=2, ensure_ascii=False)
    
    print(f"Majority baseline complete. Predicted '{majority_label}' for all {len(submission)} instances.")

if __name__ == "__main__":
    # Update paths as needed based on where you run this script
    run_baseline('../data/splits/train.json', '../data/splits/test.json', 'majority_submission.json')