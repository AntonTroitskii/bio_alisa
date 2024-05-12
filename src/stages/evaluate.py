import json

# from src.train.dataset import get_dataloaders
from pathlib import Path

import torch
from torch import nn

from mllib.pytorch.train import test_step
from mllib.utils.dvc import load_params
from mllib.utils.logs import get_logger
from mllib.utils.pkl import load_data_from_pkl
from src.train.dataset import get_data_loaders
from src.utils.files import get_label_dict


def evaluate():
    params = load_params()
    logger = get_logger("EVAL", log_level=params["base"]["log_level"])

    zip_path = Path(params["data"]["train_zip"])
    train_path = Path(params["data"]["processed_train_dir"])
    model_path = Path(params["train"]["model"])
    seed = params["base"]["seed"]
    n_images = params["train"]["number_images"]

    test_results_path = Path(params["results"]["test_results"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_dict = get_label_dict(zip_path=zip_path, prefix="", postfix=".png")

    train_dl, val_dl, test_dl = get_data_loaders(
        seed, logger, train_path, n_images, label_dict
    )

    loss_fn = nn.CrossEntropyLoss()
    model_path = Path(params["train"]["model"])

    model = load_data_from_pkl(model_path)
    test_loss, test_acc = test_step(
        model=model, dataloader=test_dl, loss_fn=loss_fn, device=device
    )

    test_results = {"test_loss": test_loss, "test_acc": test_acc}
    with open(test_results_path, "w") as res_output:
        json.dump(test_results, res_output)

    return test_results


if __name__ == "__main__":
    evaluate()
