import csv
import os
import matplotlib.pyplot as plt

def save_csv(data, filename, fieldnames):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved: {filename}")

def plot_accuracy(rounds, accuracies_dict, title, save_path):
    plt.figure(figsize=(8, 5))
    for label, accs in accuracies_dict.items():
        plt.plot(rounds, accs, marker='o', markersize=3, label=label)
    plt.xlabel('Communication round')
    plt.ylabel('Test accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")
