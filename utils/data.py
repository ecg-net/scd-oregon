from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: List[str] = None,
        first_lead_only=False,
    ):
        self.data_path = Path(data_path)
        self.split = split

        self.labels = labels
        if (self.labels is not None) and isinstance(self.labels, str):
            self.labels = [self.labels]

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        self.manifest = pd.read_csv(self.manifest_path, low_memory=False)

        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]

        self.filenames_list = list(self.manifest["filename"])

        if self.labels is not None:
            self.labels_array = self.manifest[self.labels].to_numpy()

        self.first_lead_only = first_lead_only

    def read_file(self, filepath):
        file = np.load(filepath)
        if file.shape[0] != 12:
            file = file.T
        file = torch.tensor(file).float()

        if self.first_lead_only:
            file = file[0:1]

        return file

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        filename = self.filenames_list[index]
        if self.labels is not None:
            y = self.labels_array[index]
        else:
            y = None

        filepath = self.data_path / filename
        x = self.read_file(filepath)

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if self.labels is not None and not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)
            return filename, x, y
        else:
            return filename, x
