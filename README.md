# dl_lab1

## Data

The repository includes an `images` folder that contains the visual data used in the lab. Inside `images` there is a `data` directory where all image files are stored. Also in the `images` folder is the `MAMe_dataset.csv` file which contains metadata and annotations for the specific images (for example: file names, labels, splits, and other attributes). Use `images/data/` to access the image files and `images/MAMe_dataset.csv` to load the related information.

This structure allows code to load image pixels from `images/data/` and join them with labels/attributes from `images/MAMe_dataset.csv` when preparing datasets for training and evaluation.

Note: the VGG models follow ImageNet conventions — they use stacked 3x3 convolutions (stride 1) and typically expect 224×224 inputs. The scripts in `implementations/` use 224×224 by default.

## Experiments

The `experiments.json` file defines three test configurations to demonstrate the **bias-variance tradeoff**:

### 1) Baseline Test
- **Model**: VGG16
- **Learning rate**: 0.01
- **Weight decay**: 1e-4 (light regularization)
- **Dropout**: 0.5
- **Epochs**: 15

Expected behavior: **balanced performance** — reasonable fit without overfitting.

### 2) Overfit Test
- **Model**: VGG16 (large capacity)
- **Learning rate**: 0.01
- **Weight decay**: 0.0 (no regularization)
- **Dropout**: 0.0 (no dropout)
- **Epochs**: 20 (many iterations)
- **Batch size**: 32

Expected behavior: **high variance, low bias** — model memorizes training data but generalizes poorly.

### 3) Underfit Test
- **Model**: VGG16
- **Learning rate**: 1e-6 (extremely slow — 10,000x slower than baseline)
- **Weight decay**: 1e-2 (heavy regularization)
- **Dropout**: 0.9 (very aggressive)
- **Epochs**: 15

Expected behavior: **high bias, low variance** — model severely constrained, barely learns anything even with many epochs.

## Running Experiments

Use `implementations/run_experiments.py` to train all three configurations:

```bash
python3 implementations/run_experiments.py --config implementations/experiments.json \
  --csv images/MAMe_dataset.csv --images-dir images/data --results-dir experiments_results
```

Results are logged per experiment in `experiments_results/{baseline,overfit_test,underfit_test}.log`.

## Supercomputer Run

To reproduce the experiments on the supercomputer, copy the full `dl_lab1_deploy` folder to the cluster, keep it in the same project structure as the repository, and submit `run_job.sh` from inside that directory.

The job script loads the cluster environment, runs `implementations/run_experiments.py`, and writes the outputs to `experiments_results/`. SLURM also creates `experiments_<jobid>.out` and `experiments_<jobid>.err` for the job logs.

## Local Visualization

After the job finishes, copy the `experiments_results/` folder back to the repository root so it sits outside `dl_lab1_deploy/`. Then run `analysis/visualize_results.py` locally to generate the training curves and accuracy charts from the saved `history.csv` files.

The plots are written to `analysis/plots/`.