import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    json_path = './output/model_results.json'
    if not os.path.exists(json_path):
        print(f"Error: Could not find results file at {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Configure Matplotlib for Chinese character support
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Extract data for plotting
    model_names = list(results.keys())
    accuracies = [data['accuracy'] for data in results.values()]
    train_times = [data['train_time'] for data in results.values()]

    X_axis = np.arange(len(model_names))

    # Create double-Y axis plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', color=color1, fontsize=12, fontweight='bold')
    bar1 = ax1.bar(X_axis - 0.2, accuracies, 0.4, color=color1, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 100)

    # Value labels for Accuracy
    for i, v in enumerate(accuracies):
        ax1.text(X_axis[i]-0.2, v + 2, f'{v}%', color=color1, ha='center', fontweight='bold')

    ax2 = ax1.twinx()  # Shared X axis

    color2 = 'tab:red'
    ax2.set_ylabel('Training Time (s)', color=color2, fontsize=12, fontweight='bold')
    bar2 = ax2.bar(X_axis + 0.2, train_times, 0.4, color=color2, label='Train Time')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Value labels for Train Time
    max_time = max(train_times)
    ax2.set_ylim(0, max_time * 1.2) # Give some padding
    for i, v in enumerate(train_times):
        ax2.text(X_axis[i]+0.2, v + (max_time * 0.02), f'{v}s', color=color2, ha='center', fontweight='bold')

    # Formatting Title and Legends
    plt.title('CIFAR-10 Classification Models Performance Comparison', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(X_axis)
    ax1.set_xticklabels(model_names, rotation=15)

    lines = [bar1, bar2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    
    os.makedirs('./output', exist_ok=True)
    save_path = './output/performance_comparison.png'
    plt.savefig(save_path)
    print(f"Performance chart successfully saved to {save_path}")

if __name__ == "__main__":
    main()
