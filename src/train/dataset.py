from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split


class MelDataset(Dataset):
    def __init__(self, target_dir: str, labels_dict: str, transform=None) -> None:
        super().__init__()

        self.target_dir = Path(target_dir)
        self.paths = []
        self.labels = []
        self.transform = transform

        self.paths = [target_dir / i for i in list(labels_dict.keys())]
        self.labels = list(labels_dict.values())

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:

        img = self.load_image(index)
        class_idx = self.labels[index]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


def split_train_val_test(dataset: list, fraction=[0.7, 0.2, 0.1], seed=42):
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, fraction, generator=generator)
