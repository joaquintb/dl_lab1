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
    Parse a log file and extract training loss (averaged per epoch) and validation metrics.
    
    Expected format:
        Epoch 1/5 Batch 0 train_loss 3.2345
        Epoch 1/5 val_loss 3.1234 val_acc 0.1234
    
    Returns:
        tuple: (train_losses_dict, val_losses_dict, val_accs_dict)
        where each dict is {epoch_num: value}
    """
    train_epoch_losses = {}
    train_batch_count = {}
    val_losses = {}
    val_accs = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse training loss: "Epoch 1/5 Batch 0 train_loss 3.2345"
            match_train = re.search(r'Epoch (\d+)/\d+ Batch \d+ train_loss ([\d.]+)', line)
            if match_train:
                epoch = int(match_train.group(1))
                loss = float(match_train.group(2))
                
                if epoch not in train_epoch_losses:
                    train_epoch_losses[epoch] = 0.0
                    train_batch_count[epoch] = 0
                
                train_epoch_losses[epoch] += loss
                train_batch_count[epoch] += 1
            
            # Parse validation metrics: "Epoch 1/5 val_loss 3.1234 val_acc 0.1234"
            match_val = re.search(r'Epoch (\d+)/\d+ val_loss ([\d.]+) val_acc ([\d.]+)', line)
            if match_val:
                epoch = int(match_val.group(1))
                val_loss = float(match_val.group(2))
                val_acc = float(match_val.group(3))
                
                val_losses[epoch] = val_loss
                val_accs[epoch] = val_acc
    
    # Average training losses per epoch
    avg_train_losses = {
        epoch: train_epoch_losses[epoch] / train_batch_count[epoch]
        for epoch in train_epoch_losses
    }
    
    return avg_train_losses, val_losses, val_accs


def main():
    # Paths
    results_dir = Path(__file__).parent.parent / "experiments_results"
    plots_dir = Path(__file__).parent / "plots"
    
    # Create plots directory if it doesn't exist
    plots_dir.mkdir(exist_ok=True)
    
    # Experiments to visualize
    experiments = {
        "baseline": "Baseline (balanced training, 15 epochs)",
        "overfit_test": "Overfit Test (no regularization, 25 epochs)",
        "underfit_test": "Underfit Test (heavy regularization, 5 epochs)"
    }
    
    # Parse all log files
    results = {}
    for exp_name, exp_label in experiments.items():
        log_file = results_dir / f"{exp_name}.log"
        if log_file.exists():
            train_losses, val_losses, val_accs = parse_log_file(log_file)
            results[exp_name] = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_acc': val_accs
            }
            print(f"✓ Loaded {exp_name}: {len(train_losses)} training epochs, {len(val_losses)} validation metrics")
        else:
            print(f"✗ Not found: {log_file}")
    
    if not results:
        print("No log files found. Make sure experiments_results/ exists.")
        return
    
    # === Plot 1: Comparison of all three experiments (training loss only) ===
    plt.figure(figsize=(12, 6))
    colors = {"baseline": "green", "overfit_test": "red", "underfit_test": "orange"}
    
    for exp_name, data in results.items():
        epochs = sorted(data['train_loss'].keys())
        loss_values = [data['train_loss'][e] for e in epochs]
        
        plt.plot(epochs, loss_values, 
                marker='o', linewidth=2.5, 
                label=experiments[exp_name],
                color=colors.get(exp_name, "blue"))
    
    plt.xlabel("Epoch", fontsize=12, fontweight='bold')
    plt.ylabel("Training Loss", fontsize=12, fontweight='bold')
    plt.title("VGG Training: Bias-Variance Tradeoff\n(Training Loss Comparison)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = plots_dir / "001_training_loss_comparison.png"
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    # === Plot 2-4: Individual plots for each experiment (train vs val loss) ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (exp_name, data) in enumerate(sorted(results.items())):
        ax = axes[idx]
        
        epochs_train = sorted(data['train_loss'].keys())
        train_losses = [data['train_loss'][e] for e in epochs_train]
        
        epochs_val = sorted(data['val_loss'].keys())
        val_losses = [data['val_loss'][e] for e in epochs_val]
        
        # Plot training and validation loss
        ax.plot(epochs_train, train_losses, 
               marker='o', linewidth=2.5, label='Training Loss', 
               color='steelblue')
        ax.plot(epochs_val, val_losses, 
               marker='s', linewidth=2.5, label='Validation Loss', 
               color='coral')
        
        ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')
        ax.set_ylabel("Loss", fontsize=11, fontweight='bold')
        ax.set_title(experiments[exp_name], fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = plots_dir / "002_train_vs_val_loss_subplots.png"
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    # === Plot 3: Validation accuracy comparison ===
    plt.figure(figsize=(12, 6))
    
    for exp_name, data in results.items():
        epochs = sorted(data['val_acc'].keys())
        accs = [data['val_acc'][e] * 100 for e in epochs]  # Convert to percentage
        
        plt.plot(epochs, accs, 
                marker='o', linewidth=2.5, 
                label=experiments[exp_name],
                color=colors.get(exp_name, "blue"))
    
    plt.xlabel("Epoch", fontsize=12, fontweight='bold')
    plt.ylabel("Validation Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("VGG Validation: Accuracy by Configuration", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = plots_dir / "003_validation_accuracy_comparison.png"
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    # Display summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for exp_name in sorted(results.keys()):
        data = results[exp_name]
        epochs_train = sorted(data['train_loss'].keys())
        epochs_val = sorted(data['val_loss'].keys())
        
        train_vals = [data['train_loss'][e] for e in epochs_train]
        val_vals = [data['val_loss'][e] for e in epochs_val]
        
        print(f"\n{experiments[exp_name]}:")
        print(f"  Training Loss:")
        print(f"    Initial (epoch {epochs_train[0]}): {train_vals[0]:.4f}")
        print(f"    Final (epoch {epochs_train[-1]}): {train_vals[-1]:.4f}")
        print(f"    Reduction: {train_vals[0] - train_vals[-1]:.4f}")
        
        print(f"  Validation Loss:")
        print(f"    Initial (epoch {epochs_val[0]}): {val_vals[0]:.4f}")
        print(f"    Final (epoch {epochs_val[-1]}): {val_vals[-1]:.4f}")
        
        # Calculate overfitting gap (final validation - final training)
        if epochs_train and epochs_val:
            final_train = train_vals[-1]
            final_val = val_vals[-1]
            gap = final_val - final_train
            print(f"  Overfitting Gap (Val - Train at end): {gap:.4f}")
        
        # Validation accuracy
        if data['val_acc']:
            val_accs_pct = [data['val_acc'][e] * 100 for e in epochs_val]
            print(f"  Validation Accuracy (end): {val_accs_pct[-1]:.2f}%")


if __name__ == "__main__":
    main()
