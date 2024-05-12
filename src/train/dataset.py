from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.train.models import get_grscl_transfrom


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


def get_data_loaders(seed, logger, train_path, n_images, label_dict):
    image_transform = get_grscl_transfrom()
    train_dataset = MelDataset(
        target_dir=train_path, labels_dict=label_dict, transform=image_transform
    )
    # size dimension is not how it is expected
    if n_images:
        train_dataset = Subset(train_dataset, torch.arange(0, n_images))
        logger.info("Number images to process %d", n_images)

    # Split data
    train_data, val_data, test_data = split_train_val_test(
        dataset=train_dataset, seed=seed
    )

    logger.info(
        "Train data size %d, val data size %d, test data size %d",
        len(train_data),
        len(val_data),
        len(test_data),
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=24, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        dataset=val_data, batch_size=24, shuffle=12, num_workers=0
    )

    test_dataloader = DataLoader(
        dataset=test_data, batch_size=24, shuffle=12, num_workers=0
    )

    return train_dataloader, val_dataloader, test_dataloader
