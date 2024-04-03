from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import torch
from typing import Tuple


class MelDataset(Dataset):
    def __init__(self, target_dir: str, labels_path: str, transform=None) -> None:
        super().__init__()

        self.target_dir = target_dir
        self.paths = []
        self.labels = []
        self.transform = transform

        with open(labels_path, 'r') as f:
            for line in f:
                f_name, label = line.split()
                self.paths.append(self.get_file_path(f_name))
                self.labels.append(int(label))

    def get_file_path(self, file_name, file_format='.jpg'):
        return Path(self.target_dir) / (file_name + file_format)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        
        img = self.load_image(index)
        class_idx = int(self.labels[index])
        
        
        if self.transform:
            return self.transform(self.load_image(index)), class_idx
        else:
            return self.load_image(index), class_idx
