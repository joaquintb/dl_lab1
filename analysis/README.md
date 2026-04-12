**Analysis & Visualization**

Scripts to analyze and visualize training results AFTER experiments complete.

## Usage

After running experiments and downloading `experiments_results/` to your local machine:

```bash
python analysis/visualize_results.py
```

This will:
1. Read log files from `experiments_results/` (baseline.log, overfit_test.log, underfit_test.log)
2. Parse training loss, validation loss, and accuracy per epoch for each experiment
3. Generate **three comparison plots**:
   - `001_training_loss_comparison.png` — All three training losses overlaid
   - `002_train_vs_val_loss_subplots.png` — Individual subplots showing train vs validation loss for each config
   - `003_validation_accuracy_comparison.png` — Validation accuracy curves for all three
4. Save plots to `analysis/plots/`
5. Print summary statistics for each experiment

## Expected Output Interpretation

### Plot 1: Training Loss Comparison
Shows how fast each model learns during training:
- **Baseline (green)**: Stays relatively flat ~3.37 (slow learning with lr=0.01)
- **Overfit test (red)**: Drops sharply from 3.35 → 0.05 (excellent learning, memorizing)
- **Underfit test (orange)**: Stays flat ~3.41 (learning severely inhibited)

### Plot 2: Train vs Validation Loss (Subplots)
Critical for identifying overfitting vs underfitting:
- **Baseline**: Training and validation curves track together
- **Overfit test**: Training → near 0, Validation stays ~3.30 (huge divergence = overfitting)
- **Underfit test**: Both stay high, no improvement (underfitting)

### Plot 3: Validation Accuracy
Shows model's actual classification performance on unseen data:
- **Baseline**: ~3.45% (near random chance for 29 classes)
- **Overfit test**: Reaches ~48% (learning from training data despite overfitting)
- **Underfit test**: ~3.31% (not learning)

## Summary Statistics

The script also prints key metrics:
- **Training Loss**: Initial → Final loss, reduction amount
- **Validation Loss**: Initial → Final
- **Overfitting Gap**: Difference between final validation and training loss
  - Small gap (~0) = good generalization
  - Large gap (>1.0) = overfitting
  - Negative gap = underfitting (validation better than training)
- **Validation Accuracy**: Final classification accuracy on validation set

## Requirements

- matplotlib
- numpy

Install via:
```bash
pip install matplotlib numpy
```

## Output Location

All plots are saved to `analysis/plots/`:
- `001_training_loss_comparison.png`
- `002_train_vs_val_loss_subplots.png`
- `003_validation_accuracy_comparison.png`

