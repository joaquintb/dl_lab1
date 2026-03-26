import json
import os
import argparse
from types import SimpleNamespace

from main import main as run_main


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
        ns.dropout = exp.get('dropout', 0.5)
        ns.epochs = exp.get('epochs', 1)

        # redirect prints to a per-experiment log file
        log_path = os.path.join(results_dir, f'{name}.log')
        with open(log_path, 'w') as logf:
            # simple capture by replacing stdout temporarily
            import sys
            old_stdout = sys.stdout
            sys.stdout = logf
            try:
                run_main(ns)
            except Exception as e:
                print('Experiment failed with exception:', e)
            finally:
                sys.stdout = old_stdout

        print(f'Finished {name}, log -> {log_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='implementations/experiments.json')
    p.add_argument('--csv', default='images/MAMe_dataset.csv')
    p.add_argument('--images-dir', default='images/data')
    p.add_argument('--results-dir', default='experiments_results')
    args = p.parse_args()
    run_all(args.config, args.csv, args.images_dir, args.results_dir)
