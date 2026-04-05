#!/bin/bash

#SBATCH --job-name=dl_lab1_experiments
#SBATCH --qos=debug
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=experiments_%j.out
#SBATCH --error=experiments_%j.err
#SBATCH --chdir=..

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

python3 implementations/run_experiments.py \
  --config implementations/experiments.json \
  --csv images/MAMe_dataset.csv \
  --images-dir images/data \
  --results-dir experiments_results
