"""Main training script for VGG using the local MAMe dataset loader.
"""
import argparse
import os
import sys

from model_factory import build_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from vgg_from_scratch import vgg16
from mam_dataset_loader import MAMeDataset

sys.path.append(os.path.dirname(__file__))

# Compute paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)  # parent of implementations/


def compute_metrics(model, dataloader, device, criterion):
    """Compute validation/test loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device=device, dtype=torch.long)
            else:
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy


def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load train dataset
    ds_train = MAMeDataset(args.csv, args.images_dir, transform=transform, subset='train')
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Load validation dataset
    ds_val = MAMeDataset(args.csv, args.images_dir, transform=transform, subset='val')
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load test dataset
    ds_test = MAMeDataset(args.csv, args.images_dir, transform=transform, subset='test')
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # infer num_classes from dataset if not provided
    if args.num_classes is None:
        if getattr(ds_train, 'class_to_idx', None):
            num_classes = len(ds_train.class_to_idx)
        else:
            labels = [l for l in getattr(ds_train, 'labels', []) if isinstance(l, (int,)) and l >= 0]
            if labels:
                num_classes = int(max(labels)) + 1
            else:
                raise ValueError('Could not infer number of classes from dataset; pass --num-classes')
    else:
        num_classes = args.num_classes

    model = build_model(args, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=getattr(args, 'lr', 0.01), momentum=0.9,
                                weight_decay=getattr(args, 'weight_decay', 1e-4))

    epochs = int(getattr(args, 'epochs', 1))
    
    print(f"Training on {len(ds_train)} samples, validating on {len(ds_val)}, testing on {len(ds_test)}")
    print(f"Number of classes: {num_classes}")
    print(f"\nHyperparameters:")
    print(f"  Model: {getattr(args, 'model', 'vgg16')}")
    print(f"  Learning Rate: {getattr(args, 'lr', 0.01)}")
    print(f"  Weight Decay: {getattr(args, 'weight_decay', 1e-4)}")
    print(f"  Dropout: {getattr(args, 'dropout', 0.5)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Device: {device}\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (imgs, labels) in enumerate(dl_train):
            imgs = imgs.to(device)
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device=device, dtype=torch.long)
            else:
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} Batch {batch_idx} train_loss {loss.item():.4f}')
        
        # Compute validation metrics
        val_loss, val_acc = compute_metrics(model, dl_val, device, criterion)
        print(f'Epoch {epoch+1}/{epochs} val_loss {val_loss:.4f} val_acc {val_acc:.4f}')
    
    # Compute final test metrics
    test_loss, test_acc = compute_metrics(model, dl_test, device, criterion)
    print(f'FINAL TEST RESULTS: test_loss {test_loss:.4f} test_acc {test_acc:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(_REPO_ROOT, 'images/MAMe_dataset.csv'))
    p.add_argument('--images-dir', default=os.path.join(_REPO_ROOT, 'images/data'))
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-classes', type=int, default=None,
                   help='Number of output classes. If omitted, inferred from CSV labels.')
    args = p.parse_args()
    main(args)
