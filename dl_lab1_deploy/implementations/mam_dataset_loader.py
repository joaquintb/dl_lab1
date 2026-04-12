import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class MAMeDataset(Dataset):
    """
    Dataset loader for the MAMe CSV schema.

    Expected columns:
    - Image file
    - Medium
    - Subset
    """

    def __init__(self, csv_path, images_dir, transform=None, subset=None):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.transform = transform

        df = pd.read_csv(csv_path)

        required_cols = ["Image file", "Medium", "Subset"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Build a global label mapping BEFORE filtering by subset
        unique_labels = sorted(df["Medium"].dropna().astype(str).unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {i: label for label, i in self.class_to_idx.items()}

        if subset is not None:
            subset_map = {
                "train": "train",
                "training": "train",
                "val": "val",
                "validation": "val",
                "test": "test",
            }

            subset_norm = subset_map.get(str(subset).strip().lower(), str(subset).strip().lower())
            csv_subset_norm = df["Subset"].astype(str).str.strip().str.lower().map(
                lambda x: subset_map.get(x, x)
            )

            df = df[csv_subset_norm == subset_norm].reset_index(drop=True)

            if len(df) == 0:
                available_subsets = sorted(df["Subset"].astype(str).str.strip().str.lower().unique())
                raise ValueError(
                    f"No rows found for subset='{subset}'. "
                    f"Available subset values may include: {available_subsets}"
                )

        self.filenames = df["Image file"].astype(str).tolist()
        self.labels = [self.class_to_idx[str(v)] for v in df["Medium"].astype(str).tolist()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.images_dir, fname)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label