"""
Visualize training results from experiments_results/ logs.
Generates loss curves for baseline, overfit_test, and underfit_test.
"""

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """
    Parse a log file and extract average loss per epoch.
    
    Expected format:
        Epoch 1/5 Batch 0 loss 3.2345
        Epoch 1/5 Batch 10 loss 3.1234
        ...
    
    Returns:
        dict: {epoch_num: avg_loss}
    """
    epoch_losses = {}
    batch_count = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse: "Epoch 1/5 Batch 0 loss 3.2345"
            match = re.search(r'Epoch (\d+)/\d+ Batch \d+ loss ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                
                # Accumulate losses per epoch
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = 0.0
                    batch_count[epoch] = 0
                
                epoch_losses[epoch] += loss
                batch_count[epoch] += 1
    
    # Average losses per epoch
    avg_losses = {
        epoch: epoch_losses[epoch] / batch_count[epoch]
        for epoch in epoch_losses
    }
    
    return avg_losses


def main():
    # Paths
    results_dir = Path(__file__).parent.parent / "experiments_results"
    plots_dir = Path(__file__).parent / "plots"
    
    # Create plots directory if it doesn't exist
    plots_dir.mkdir(exist_ok=True)
    
    # Experiments to visualize
    experiments = {
        "baseline": "Baseline (balanced)",
        "overfit_test": "Overfit Test (no regularization, 20 epochs)",
        "underfit_test": "Underfit Test (heavy regularization, lr=1e-5)"
    }
    
    # Parse all log files
    results = {}
    for exp_name, exp_label in experiments.items():
        log_file = results_dir / f"{exp_name}.log"
        if log_file.exists():
            results[exp_name] = parse_log_file(log_file)
            print(f"✓ Loaded {exp_name}: {len(results[exp_name])} epochs")
        else:
            print(f"✗ Not found: {log_file}")
    
    if not results:
        print("No log files found. Make sure experiments_results/ exists.")
        return
    
    # Create figure with loss curves
    plt.figure(figsize=(12, 6))
    
    colors = {"baseline": "green", "overfit_test": "red", "underfit_test": "orange"}
    
    for exp_name, losses in results.items():
        epochs = sorted(losses.keys())
        loss_values = [losses[e] for e in epochs]
        
        plt.plot(epochs, loss_values, 
                marker='o', linewidth=2.5, 
                label=experiments[exp_name],
                color=colors.get(exp_name, "blue"))
    
    plt.xlabel("Epoch", fontsize=12, fontweight='bold')
    plt.ylabel("Training Loss", fontsize=12, fontweight='bold')
    plt.title("VGG Training: Bias-Variance Tradeoff\n(Baseline vs Overfit vs Underfit)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = plots_dir / "training_loss_comparison.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\n✓ Saved plot: {plot_file}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for exp_name, losses in results.items():
        epochs = sorted(losses.keys())
        loss_vals = [losses[e] for e in epochs]
        print(f"\n{experiments[exp_name]}:")
        print(f"  Final loss (epoch {epochs[-1]}): {loss_vals[-1]:.4f}")
        print(f"  Initial loss (epoch {epochs[0]}): {loss_vals[0]:.4f}")
        print(f"  Loss reduction: {loss_vals[0] - loss_vals[-1]:.4f}")
        print(f"  Min loss: {min(loss_vals):.4f} (epoch {epochs[loss_vals.index(min(loss_vals))]})")


if __name__ == "__main__":
    main()
