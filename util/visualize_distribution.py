import matplotlib.pyplot as plt
import os

# Data based on previous script results
labels = ['ADVICE_SEEKING', 'PERSONAL_EXPERIENCE', 'OPINION', 'OTHER']
counts = [673, 618, 544, 321]

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color=['skyblue', 'salmon', 'lightgreen', 'orange'])
plt.xlabel('Post Intent Category', fontsize=12)
plt.ylabel('Number of Instances', fontsize=12)
plt.title('Distribution of Social Media Post Intents', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add count labels on top of bars
for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha='center', fontweight='bold')

os.makedirs("results", exist_ok=True)
plt.savefig("results/label_distribution.png")
print("Saved bar chart to results/label_distribution.png")
