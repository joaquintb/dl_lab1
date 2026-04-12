import json
import os
import argparse
import traceback
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace

from main import main as run_main

# Compute paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


def build_args_namespace(exp, csv, images_dir, output_dir):
    ns = SimpleNamespace()

    # Shared paths
    ns.csv = csv
    ns.images_dir = images_dir
    ns.output_dir = output_dir
    ns.experiment_name = exp.get("name", "exp")

    # Experiment config with defaults
    ns.batch_size = exp.get("batch_size", 16)
    ns.num_classes = exp.get("num_classes", None)

    ns.model = exp.get("model", "vgg16")
    ns.lr = exp.get("lr", 0.01)
    ns.weight_decay = exp.get("weight_decay", 1e-4)
    ns.dropout = exp.get("dropout", None)
    ns.epochs = exp.get("epochs", 1)

    ns.augmentation = exp.get("augmentation", "none")
    ns.image_size = exp.get("image_size", 224)
    ns.num_workers = exp.get("num_workers", 2)
    ns.log_every = exp.get("log_every", 10)

    return ns


def run_all(config_path, csv, images_dir, results_dir="experiments_results"):
    os.makedirs(results_dir, exist_ok=True)

    with open(config_path, "r") as f:
        experiments = json.load(f)

    for i, exp in enumerate(experiments):
        name = exp.get("name", f"exp_{i:03d}")
        experiment_dir = os.path.join(results_dir, name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Save experiment config used for this run
        config_save_path = os.path.join(experiment_dir, "config.json")
        with open(config_save_path, "w") as f:
            json.dump(exp, f, indent=2)

        ns = build_args_namespace(
            exp=exp,
            csv=csv,
            images_dir=images_dir,
            output_dir=experiment_dir,
        )

        log_path = os.path.join(experiment_dir, "run.log")

        print(f"=== Running experiment: {name} ===")
        print(f"Output folder: {experiment_dir}")

        with open(log_path, "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            try:
                print(f"EXPERIMENT {name}")
                print(
                    f"CONFIG "
                    f"model={ns.model} "
                    f"lr={ns.lr} "
                    f"weight_decay={ns.weight_decay} "
                    f"dropout={ns.dropout} "
                    f"epochs={ns.epochs} "
                    f"batch_size={ns.batch_size} "
                    f"augmentation={ns.augmentation} "
                    f"image_size={ns.image_size} "
                    f"num_workers={ns.num_workers} "
                    f"log_every={ns.log_every}"
                )
                run_main(ns)
            except Exception as e:
                print(f"Experiment failed with exception: {e}")
                traceback.print_exc()

        print(f"Finished {name}, log -> {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(_REPO_ROOT, "implementations/experiments.json"))
    p.add_argument("--csv", default=os.path.join(_REPO_ROOT, "images/MAMe_dataset.csv"))
    p.add_argument("--images-dir", default=os.path.join(_REPO_ROOT, "images/data"))
    p.add_argument("--results-dir", default=os.path.join(_REPO_ROOT, "experiments_results"))
    args = p.parse_args()

    run_all(args.config, args.csv, args.images_dir, args.results_dir)