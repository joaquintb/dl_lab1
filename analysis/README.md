**Analysis & Visualization**

Scripts to analyze and visualize training results AFTER experiments complete.

## Usage

After running experiments and downloading `experiments_results/` to your local machine:

```bash
python analysis/visualize_results.py
```

This will:
1. Read log files from `experiments_results/`
2. Parse loss values per epoch for each experiment
3. Generate a comparison plot of all three experiments
4. Save the plot to `analysis/plots/training_loss_comparison.png`
5. Print summary statistics (final loss, loss reduction, min loss)

## Expected Output

The plot should show:
- **Baseline (green)**: Moderate loss decrease, balanced learning
- **Overfit test (red)**: Loss drops very low (overfitting), no regularization
- **Underfit test (orange)**: Loss decreases slowly, heavy regularization prevents learning

## Requirements

- matplotlib
- numpy

Install via:
```bash
pip install matplotlib numpy
```
