import json
import os
import argparse
from types import SimpleNamespace
import sys

from main import main as run_main

# Compute paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


def run_all(config_path, csv, images_dir, results_dir='experiments_results'):
    os.makedirs(results_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        experiments = json.load(f)

    for exp in experiments:
        name = exp.get('name', 'exp')
        print(f'=== Running experiment: {name} ===')

        ns = SimpleNamespace()
        ns.csv = csv
        ns.images_dir = images_dir

        ns.batch_size = exp.get('batch_size', 16)
        ns.num_classes = exp.get('num_classes', None)

        ns.model = exp.get('model', 'vgg16')
        ns.lr = exp.get('lr', 0.01)
        ns.weight_decay = exp.get('weight_decay', 1e-4)
        ns.dropout = exp.get('dropout', None)
        ns.epochs = exp.get('epochs', 1)

        ns.augmentation = exp.get('augmentation', 'none')
        ns.image_size = exp.get('image_size', 224)
        ns.num_workers = exp.get('num_workers', 2)
        ns.log_every = exp.get('log_every', 10)

        log_path = os.path.join(results_dir, f'{name}.log')

        with open(log_path, 'w') as logf:
            old_stdout = sys.stdout
            sys.stdout = logf
            try:
                print(f'EXPERIMENT {name}')
                print(
                    f'CONFIG '
                    f'model={ns.model} '
                    f'lr={ns.lr} '
                    f'weight_decay={ns.weight_decay} '
                    f'dropout={ns.dropout} '
                    f'epochs={ns.epochs} '
                    f'batch_size={ns.batch_size} '
                    f'augmentation={ns.augmentation} '
                    f'image_size={ns.image_size} '
                    f'num_workers={ns.num_workers} '
                    f'log_every={ns.log_every}'
                )
                run_main(ns)
            except Exception as e:
                print('Experiment failed with exception:', e)
            finally:
                sys.stdout = old_stdout

        print(f'Finished {name}, log -> {log_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(_REPO_ROOT, 'implementations/experiments.json'))
    p.add_argument('--csv', default=os.path.join(_REPO_ROOT, 'images/MAMe_dataset.csv'))
    p.add_argument('--images-dir', default=os.path.join(_REPO_ROOT, 'images/data'))
    p.add_argument('--results-dir', default=os.path.join(_REPO_ROOT, 'experiments_results'))
    args = p.parse_args()
    run_all(args.config, args.csv, args.images_dir, args.results_dir)