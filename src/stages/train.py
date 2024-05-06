import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from mllib.pytorch.files import save_results_as_csv
from mllib.pytorch.plot import save_plot_results
from mllib.pytorch.train import train
from mllib.utils.dvc import load_params
from mllib.utils.io import make_folder
from mllib.utils.logs import get_logger
from mllib.utils.pkl import save_data_as_pkl
from src.train.dataset import MelDataset, split_train_val_test
from src.train.models import get_grscl_transfrom, get_torch_model
from src.utils.files import get_label_dict


def train_model():
    params = load_params()
    logger = get_logger("TRAIN", log_level=params["base"]["log_level"])

    zip_path = Path(params["data"]["train_zip"])
    train_path = Path(params["data"]["processed_train_dir"])
    model_path = Path(params["train"]["model"])
    n_images = params["train"]["number_images"]

    resluts_path_csv = Path(params["results"]["train_results_csv"])
    resluts_path_png = Path(params["results"]["train_results_png"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_transform = get_grscl_transfrom()

    label_dict = get_label_dict(zip_path=zip_path, prefix="", postfix=".png")
    train_dataset = MelDataset(
        target_dir=train_path, labels_dict=label_dict, transform=image_transform
    )
    # size dimension is not how it is expected
    if n_images:
        train_dataset = Subset(train_dataset, torch.arange(0, n_images))
        logger.info("Number images to process {}".format(n_images))

    # Split data
    train_data, val_data, test_data = split_train_val_test(
        dataset=train_dataset, seed=params["base"]["seed"]
    )

    logger.info(
        "Train data size {}, val data size {}".format(len(train_data), len(val_data))
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=24, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        dataset=val_data, batch_size=24, shuffle=12, num_workers=0
    )
    lr = params["train"]["lr"]
    epochs = params["train"]["epochs"]

    model = get_torch_model(params=params, device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("TRAIN STARTS ...")
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        epochs=epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )
    logger.info("TRAIN COMPLEATES.")

    logger.info("Current directory {}".format(os.getcwd()))
    make_folder(model_path.parent)
    save_data_as_pkl(model, model_path)

    make_folder(resluts_path_csv.parent)
    save_results_as_csv(results=results, file_path=resluts_path_csv)
    save_plot_results(results=results, file_path=resluts_path_png)

    return results


if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('--config', dest='config', required=True)
    # args = arg_parser.parse_args()

    # train_model(config_path=args.config)

    train_model()
