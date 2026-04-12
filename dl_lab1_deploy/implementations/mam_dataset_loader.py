import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class MAMeDataset(Dataset):
    """
    CSV-driven dataset loader for the MAMe dataset in `images/data/`.

    Expects a CSV with at least a filename column and a label column.
    By default it will try common column names;
    you can override with `filename_col` and `label_col`.
    """

    def __init__(self, csv_path, images_dir, transform=None, filename_col=None,
                 label_col=None, subset=None):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        
        # Filter by subset
        if subset is not None:
            if 'Subset' in df.columns:
                df = df[df['Subset'] == subset].reset_index(drop=True)
            else:
                raise ValueError(f"'Subset' column not found in CSV. Available columns: {df.columns.tolist()}")

        def _norm(s):
            return ''.join(c for c in str(s).lower() if c.isalnum())

        col_map = {_norm(c): c for c in df.columns}

        if filename_col is None:
            for cand in ['filename', 'file', 'image', 'img', 'imagefile']:
                if cand in col_map:
                    filename_col = col_map[cand]
                    break
        if label_col is None:
            for cand in ['label', 'class', 'target', 'medium', 'category']:
                if cand in col_map:
                    label_col = col_map[cand]
                    break

        if filename_col is None:
            raise ValueError('Could not find a filename column in CSV; pass filename_col')
        if label_col is None:
            df['__label__'] = -1
            label_col = '__label__'

        self.filenames = df[filename_col].astype(str).tolist()
        raw_labels = df[label_col].tolist()

        try:
            if df[label_col].dtype == object or any(isinstance(l, str) for l in raw_labels):
                uniques = []
                seen = set()
                for v in raw_labels:
                    if pd.isna(v):
                        continue
                    if v not in seen:
                        seen.add(v)
                        uniques.append(v)
                self.class_to_idx = {v: i for i, v in enumerate(uniques)}
                self.idx_to_class = {i: v for v, i in self.class_to_idx.items()}
                self.labels = [self.class_to_idx.get(v, -1) if not pd.isna(v) else -1 for v in raw_labels]
            else:
                self.class_to_idx = None
                self.idx_to_class = None
                self.labels = [int(v) if not pd.isna(v) else -1 for v in raw_labels]
        except Exception:
            self.class_to_idx = None
            self.idx_to_class = None
            self.labels = raw_labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    from torchvision import transforms
    csv = os.path.join('images', 'MAMe_dataset.csv')
    imgs = os.path.join('images', 'data')
    
    # Test with train subset
    ds_train = MAMeDataset(csv, imgs, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), subset='train')
    print('Train dataset size:', len(ds_train))

    # Test with validation subset
    ds_val = MAMeDataset(csv, imgs, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), subset='validation')
    print('Validation dataset size:', len(ds_val))

    # Test with test subset
    ds_test = MAMeDataset(csv, imgs, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), subset='test')
    print('Test dataset size:', len(ds_test))

    # Test loading a sample
    x, y = ds_train[-1]
    print('Sample shape:', x.shape, 'label:', y)
