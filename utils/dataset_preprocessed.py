import torch
import json
from torch.utils.data import Dataset
import numpy as np
import os

class ClimatehackDatasetPreprocessed(Dataset):
    def __init__(self, data_path, metdata_path) -> None:
        super().__init__()

        with open(metdata_path) as f:
            self.filename_list = json.load(f)

        self.data_path = data_path

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        filename = self.filename_list[index]
        x = np.load(os.path.join(self.data_path, filename + '_input.npy'))
        y = np.load(os.path.join(self.data_path, filename + '_target.npy'))

        # create channel dimension
        x = x[:, np.newaxis, :, :].astype(np.float32) / 1024
        y = y[:, np.newaxis, 32: 96, 32: 96].astype(np.float32)

        return x, y
