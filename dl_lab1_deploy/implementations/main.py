"""Main training script for VGG using the local MAMe dataset loader.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from vgg_from_scratch import vgg16, vgg11
from mam_dataset_loader import MAMeDataset

sys.path.append(os.path.dirname(__file__))

# Compute paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)  # parent of implementations/


def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = MAMeDataset(args.csv, args.images_dir, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # infer num_classes from dataset if not provided
    if args.num_classes is None:
        if getattr(ds, 'class_to_idx', None):
            num_classes = len(ds.class_to_idx)
        else:
            labels = [l for l in getattr(ds, 'labels', []) if isinstance(l, (int,)) and l >= 0]
            if labels:
                num_classes = int(max(labels)) + 1
            else:
                raise ValueError('Could not infer number of classes from dataset; pass --num-classes')
    else:
        num_classes = args.num_classes

    # build model (support vgg11/vgg16)
    if getattr(args, 'model', None) == 'vgg11':
        model = vgg11(num_classes=num_classes).to(device)
    else:
        model = vgg16(num_classes=num_classes).to(device)

    # adjust dropout in classifier if requested
    if getattr(args, 'dropout', None) is not None:
        for m in model.classifier.modules():
            if isinstance(m, nn.Dropout):
                m.p = float(args.dropout)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=getattr(args, 'lr', 0.01), momentum=0.9,
                                weight_decay=getattr(args, 'weight_decay', 1e-4))

    epochs = int(getattr(args, 'epochs', 1))
    model.train()
    for epoch in range(epochs):
        for batch_idx, (imgs, labels) in enumerate(dl):
            imgs = imgs.to(device)
            # labels may already be a tensor (default_collate) or a list; handle both without copying tensors
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device=device, dtype=torch.long)
            else:
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} Batch {batch_idx} loss {loss.item():.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(_REPO_ROOT, 'images/MAMe_dataset.csv'))
    p.add_argument('--images-dir', default=os.path.join(_REPO_ROOT, 'images/data'))
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-classes', type=int, default=None,
                   help='Number of output classes. If omitted, inferred from CSV labels.')
    args = p.parse_args()
    main(args)
