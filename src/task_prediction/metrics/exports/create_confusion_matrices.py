import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm
from typing import Any
from pathlib import Path

def clean_label(label: str):
    """Converts labels like 'ENTRY_CONFLICT_RESOLUTION' to 'Entry Conflict Resolution'."""
    return label.replace('_', ' ').title()

def plot_confusion_matrix(cm, classes: list[str], title: str, filename: str) -> None:
    """
    Generates a highly-polished confusion matrix heatmap with italicized percentage labels.
    """
    cm_array = np.array(cm)
    total_samples = cm_array.sum()
    
    # Calculate marginal distributions (sums)
    true_sums = cm_array.sum(axis=1)  # Row sums
    pred_sums = cm_array.sum(axis=0)  # Column sums
    
    # Create multi-line labels: Class name on top, italicized percentage on bottom
    # We use Matplotlib's mathtext $\mathit{...}$ to force italics on the percentage
    y_labels = [
        f"{clean_label(cls)}\n$\\mathit{{({val/total_samples*100:.1f}\\%)}}$" 
        for cls, val in zip(classes, true_sums)
    ]
    x_labels = [
        f"{clean_label(cls)}\n$\\mathit{{({val/total_samples*100:.1f}\\%)}}$" 
        for cls, val in zip(classes, pred_sums)
    ]

    is_large_matrix = len(classes) > 5

    # DYNAMIC FIGURE SIZING
    if is_large_matrix:
        # Give large matrices plenty of room
        fig_size = max(12, len(classes) * 1.3)
        _, ax = plt.subplots(figsize=(fig_size + 3, fig_size))
    else:
        # Tighter bounding box for the 2x2 matrix
        _, ax = plt.subplots(figsize=(10, 9))
    
    # Create custom annotations (Hide 0s to declutter sparse matrices)
    annot_matrix = np.empty_like(cm_array, dtype=object)
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            val = cm_array[i, j]
            if val == 0:
                annot_matrix[i, j] = "" # Hide zeros completely
            else:
                annot_matrix[i, j] = f"{val:,}" # Add comma formatting

    # Use Symmetrical Log Normalization to handle massive imbalances
    norm = SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=cm_array.max(), base=10)

    # DYNAMIC FONT SIZING based on matrix size
    annot_size = 18 if is_large_matrix else 32
    tick_size = 16 if is_large_matrix else 20

    # Plot the heatmap
    heatmap = sns.heatmap(
        cm_array, 
        annot=annot_matrix, 
        fmt="", 
        cmap="Blues", 
        cbar=True,
        cbar_kws={'shrink': 0.8},
        norm=norm, 
        linewidths=0.5, 
        linecolor='lightgray',
        xticklabels=x_labels, 
        yticklabels=y_labels,
        ax=ax,
        square=True,  # Forces perfect squares
        annot_kws={"size": annot_size, "weight": "medium"} 
    )

    # Format Colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Count (Log Scale)', size=18, weight='bold', labelpad=15)

    # Styling the axes titles
    ax.set_xlabel('Predicted Class Distribution', fontsize=24, fontweight='bold', labelpad=25)
    ax.set_ylabel('True Class Distribution', fontsize=24, fontweight='bold', labelpad=25)
    
    # Tick label formatting (Aligning the two lines beautifully)
    plt.xticks(rotation=45, ha='right', fontsize=tick_size)
    plt.yticks(rotation=0, fontsize=tick_size, va='center')

    # Main Title
    plt.title(title, fontsize=32, fontweight='bold', pad=30)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

def create_global_cms(metrics: dict[str, Any], output_path: Path) -> None:
    plot_confusion_matrix(
        cm=metrics['stage_1']['cm'],
        classes=metrics['stage_1']['classes'],
        title="Stage 1 Confusion Matrix",
        filename=output_path/"global_cm_stage_1.png"
    )

    plot_confusion_matrix(
        cm=metrics['stage_2']['cm'],
        classes=metrics['stage_2']['classes'],
        title="Stage 2 Confusion Matrix",
        filename=output_path/"global_cm_stage_2.png"
    )

    plot_confusion_matrix(
        cm=metrics['joint']['cm'],
        classes=metrics['joint']['classes'],
        title="Joint Model Confusion Matrix",
        filename=output_path/"global_cm_joint.png"
    )

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Generate Confusion Matrices from JSON data.")
    parser.add_argument("json_file", type=str, help="Path to the input JSON file.")
    args = parser.parse_args()

    # Load JSON Data
    try:
        with open(args.json_file, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{args.json_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{args.json_file}' is not valid JSON.")
        return

    print("Generating matrices with italicized multi-line labels...")
    create_global_cms(metrics)

if __name__ == "__main__":
    main()