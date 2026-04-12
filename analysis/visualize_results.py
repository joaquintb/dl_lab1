"""
Visualize train vs validation loss from experiment log files.
Generates a single subplot figure with one panel per experiment log.
"""

import re
from pathlib import Path
import math
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    """
    Parse a log file and extract:
    - average training loss per epoch (from batch logs)
    - validation loss per epoch

    Expected lines:
        Epoch 1/5 Batch 0 train_loss 3.2345
        Epoch 1/5 val_loss 3.1234 val_acc 0.1234
    """
    train_epoch_losses = {}
    train_batch_count = {}
    val_losses = {}

    with open(log_path, "r") as f:
        for line in f:
            match_train = re.search(r"Epoch (\d+)/\d+ Batch \d+ train_loss ([\d.]+)", line)
            if match_train:
                epoch = int(match_train.group(1))
                loss = float(match_train.group(2))

                if epoch not in train_epoch_losses:
                    train_epoch_losses[epoch] = 0.0
                    train_batch_count[epoch] = 0

                train_epoch_losses[epoch] += loss
                train_batch_count[epoch] += 1

            match_val = re.search(r"Epoch (\d+)/\d+ val_loss ([\d.]+) val_acc ([\d.]+)", line)
            if match_val:
                epoch = int(match_val.group(1))
                val_loss = float(match_val.group(2))
                val_losses[epoch] = val_loss

    avg_train_losses = {
        epoch: train_epoch_losses[epoch] / train_batch_count[epoch]
        for epoch in train_epoch_losses
        if train_batch_count[epoch] > 0
    }

    return avg_train_losses, val_losses


def make_pretty_title(log_path):
    return log_path.stem.replace("_", " ").title()


def main():
    results_dir = Path(__file__).parent.parent / "experiments_results"
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    log_files = sorted(results_dir.glob("*.log"))

    if not log_files:
        print("No log files found in experiments_results/")
        return

    results = []
    for log_file in log_files:
        train_losses, val_losses = parse_log_file(log_file)

        if not train_losses and not val_losses:
            print(f"Skipping {log_file.name}: no parsable training/validation loss found.")
            continue

        results.append({
            "name": log_file.stem,
            "title": make_pretty_title(log_file),
            "train_loss": train_losses,
            "val_loss": val_losses,
        })
        print(f"Loaded {log_file.name}")

    if not results:
        print("No usable log data found.")
        return

    n = len(results)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, result in zip(axes, results):
        epochs_train = sorted(result["train_loss"].keys())
        train_losses = [result["train_loss"][e] for e in epochs_train]

        epochs_val = sorted(result["val_loss"].keys())
        val_losses = [result["val_loss"][e] for e in epochs_val]

        if epochs_train:
            ax.plot(
                epochs_train,
                train_losses,
                marker="o",
                linewidth=2.0,
                label="Training Loss",
            )

        if epochs_val:
            ax.plot(
                epochs_val,
                val_losses,
                marker="s",
                linewidth=2.0,
                label="Validation Loss",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(result["title"])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplot axes
    for ax in axes[len(results):]:
        ax.axis("off")

    plt.tight_layout()

    plot_file = plots_dir / "train_vs_val_loss_subplots.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Saved: {plot_file}")
    plt.close()


if __name__ == "__main__":
    main()