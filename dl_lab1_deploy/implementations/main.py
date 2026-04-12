"""Main training script for one experiment on the MAMe dataset."""
import copy
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_factory import build_model
from mam_dataset_loader import MAMeDataset
from transforms_factory import build_transforms
from train_eval_utils import (
    train_one_epoch,
    validate,
    test_model,
    save_history_csv,
    save_test_metrics_json,
)

sys.path.append(os.path.dirname(__file__))

# Compute paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)  # parent of implementations/


def build_dataloaders(args):
    train_transform, eval_transform = build_transforms(
        augmentation=getattr(args, "augmentation", "none"),
        image_size=getattr(args, "image_size", 224),
    )

    num_workers = getattr(args, "num_workers", 2)

    ds_train = MAMeDataset(args.csv, args.images_dir, transform=train_transform, subset="train")
    ds_val = MAMeDataset(args.csv, args.images_dir, transform=eval_transform, subset="val")
    ds_test = MAMeDataset(args.csv, args.images_dir, transform=eval_transform, subset="test")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test


def main(args):
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    ds_train, ds_val, ds_test, dl_train, dl_val, dl_test = build_dataloaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 29

    model = build_model(args, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=getattr(args, "lr", 0.01),
        momentum=0.9,
        weight_decay=getattr(args, "weight_decay", 1e-4),
    )

    epochs = int(getattr(args, "epochs", 1))
    log_every = int(getattr(args, "log_every", 10))

    print(
        f"RUN_INFO "
        f"model={getattr(args, 'model', 'vgg16')} "
        f"lr={getattr(args, 'lr', 0.01)} "
        f"weight_decay={getattr(args, 'weight_decay', 1e-4)} "
        f"dropout={getattr(args, 'dropout', None)} "
        f"epochs={epochs} "
        f"batch_size={args.batch_size} "
        f"augmentation={getattr(args, 'augmentation', 'none')} "
        f"image_size={getattr(args, 'image_size', 224)} "
        f"device={device}"
    )
    print(
        f"DATA_INFO "
        f"train_samples={len(ds_train)} "
        f"val_samples={len(ds_val)} "
        f"test_samples={len(ds_test)} "
        f"num_classes={num_classes}"
    )

    best_val_acc = -1.0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dl_train,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            epochs=epochs,
            log_every=log_every,
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=dl_val,
            device=device,
            criterion=criterion,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"EPOCH_SUMMARY "
            f"epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

    print(f"BEST_CHECKPOINT epoch={best_epoch} val_acc={best_val_acc:.4f}")

    model.load_state_dict(best_state_dict)

    if output_dir is not None:
        # checkpoint_path = os.path.join(output_dir, "best_model.pt")
        # torch.save(
        #     {
        #         "epoch": best_epoch,
        #         "val_acc": best_val_acc,
        #         "model_state_dict": model.state_dict(),
        #     },
        #     checkpoint_path,
        # )
        save_history_csv(history_rows, output_dir)

    test_loss, test_acc, macro_f1, weighted_f1, macro_precision, macro_recall = test_model(
        model=model,
        dataloader=dl_test,
        device=device,
        criterion=criterion,
    )

    test_metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }

    if output_dir is not None:
        save_test_metrics_json(test_metrics, output_dir)

    print(
        f"FINAL_TEST "
        f"best_epoch={best_epoch} "
        f"best_val_acc={best_val_acc:.4f} "
        f"test_loss={test_loss:.4f} "
        f"test_acc={test_acc:.4f} "
        f"macro_f1={macro_f1:.4f} "
        f"weighted_f1={weighted_f1:.4f} "
        f"macro_precision={macro_precision:.4f} "
        f"macro_recall={macro_recall:.4f}"
    )
