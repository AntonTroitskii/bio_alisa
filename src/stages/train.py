import os
from pathlib import Path

import torch
from torch import nn

from mllib.pytorch.files import save_results_as_csv
from mllib.pytorch.plot import save_plot_results
from mllib.pytorch.train import train
from mllib.utils.dvc import load_params
from mllib.utils.io import make_folder
from mllib.utils.logs import get_logger
from mllib.utils.pkl import save_data_as_pkl
from src.train.dataset import get_data_loaders
from src.train.models import get_torch_model
from src.utils.files import get_label_dict


def train_model():
    params = load_params()

    logger = get_logger("TRAIN", log_level=params["base"]["log_level"])

    zip_path = Path(params["data"]["train_zip"])
    train_path = Path(params["data"]["processed_train_dir"])
    model_path = Path(params["train"]["model"])
    seed = params["base"]["seed"]
    n_images = params["train"]["number_images"]
    resluts_path_csv = Path(params["results"]["train_results_csv"])
    resluts_path_png = Path(params["results"]["train_results_png"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_dict = get_label_dict(zip_path=zip_path, prefix="", postfix=".png")
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(
        seed, logger, train_path, n_images, label_dict
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
