import json
from collections import Counter
import matplotlib.pyplot as plt
import os

def create_visualization(input_path, output_path):
    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count labels
    counts = Counter(d['label'] for d in data)
    
    # Define labels and their counts (sorted for consistency)
    labels = sorted(counts.keys())
    values = [counts[label] for label in labels]
    
    # Colors for each label
    color_map = {
        'ADVICE_SEEKING': 'skyblue',
        'PERSONAL_EXPERIENCE': 'salmon',
        'OPINION': 'lightgreen',
        'OTHER': 'orange'
    }
    colors = [color_map.get(label, 'grey') for label in labels]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    
    # Labels and title
    plt.xlabel('Post Intent Category', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.title('Label Distribution in Gold Standard Dataset', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    input_file = "data/final_data/gold_standard.json"
    output_file = "results/gold_standard_distribution.png"
    create_visualization(input_file, output_file)
