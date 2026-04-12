from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_history(history_path):
    df = pd.read_csv(history_path)

    required_cols = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{history_path} is missing required columns: {sorted(missing)}"
        )

    return df.sort_values("epoch").reset_index(drop=True)


def plot_loss(df, experiment_name, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="s", linewidth=2, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{experiment_name} - Train vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy(df, experiment_name, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", linewidth=2, label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], marker="s", linewidth=2, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    analysis_dir = Path(__file__).resolve().parent
    results_dir = analysis_dir.parent / "experiments_results"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    experiment_dirs = sorted(
        [p for p in results_dir.iterdir() if p.is_dir()]
    )

    if not experiment_dirs:
        print(f"No experiment folders found in: {results_dir}")
        return

    for exp_dir in experiment_dirs:
        history_path = exp_dir / "history.csv"

        if not history_path.exists():
            print(f"Skipping {exp_dir.name}: no history.csv found")
            continue

        try:
            df = load_history(history_path)

            loss_path = plots_dir / f"{exp_dir.name}_loss.png"
            acc_path = plots_dir / f"{exp_dir.name}_accuracy.png"

            plot_loss(df, exp_dir.name, loss_path)
            plot_accuracy(df, exp_dir.name, acc_path)

            print(f"Saved: {loss_path}")
            print(f"Saved: {acc_path}")

        except Exception as e:
            print(f"Failed for {exp_dir.name}: {e}")


if __name__ == "__main__":
    main()