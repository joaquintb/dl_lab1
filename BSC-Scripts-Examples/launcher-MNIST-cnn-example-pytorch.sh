#!/bin/bash

###

#SBATCH --qos=debug

###

#SBATCH --cpus-per-task=40

#SBATCH --gres gpu:1

#SBATCH --time=02:00:00

###

#SBATCH --job-name="test_mnist-pytorch"

#SBATCH --chdir=.

#SBATCH --output=test_mnist-pytorch_%j.out

#SBATCH --error=test_mnist-pytorch_%j.err

###

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

python MNIST-cnn-example-pytorch.py

