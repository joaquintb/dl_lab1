import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def _move_labels_to_device(labels, device):
    if isinstance(labels, torch.Tensor):
        return labels.to(device=device, dtype=torch.long)
    return torch.tensor(labels, dtype=torch.long, device=device)


def train_one_epoch(model, dataloader, device, criterion, optimizer, epoch, epochs, log_every=10):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = _move_labels_to_device(labels, device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size

        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        if batch_idx % log_every == 0:
            print(f'Epoch {epoch}/{epochs} Batch {batch_idx} train_loss {loss.item():.4f}')

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


def validate(model, dataloader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = _move_labels_to_device(labels, device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


def test_model(model, dataloader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = _move_labels_to_device(labels, device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, avg_acc, macro_f1, weighted_f1, macro_precision, macro_recall