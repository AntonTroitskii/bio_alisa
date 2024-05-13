from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from mllib.pytorch.files import save_results_as_csv
from mllib.pytorch.train import train_to_predict
from mllib.utils.dvc import load_params
from mllib.utils.io import make_folder
from mllib.utils.logs import get_logger
from mllib.utils.pkl import save_data_as_pkl
from src.train.dataset import MelDataset, get_grscl_transfrom
from src.train.models import get_torch_model
from src.utils.files import get_label_dict


def train_to_predict_stage():

    logger = get_logger(name="TRAIN OT PREDICT")
    params = load_params()

    zip_path = Path(params["data"]["train_zip"])
    train_dir = Path(params["data"]["processed_train_dir"])
    train_predict_path_csv = Path(params["results"]["train_predict_results_csv"])

    model_path = Path(params["predict"]["model"])

    n_images = params["train"]["number_images"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_transform = get_grscl_transfrom()
    label_dict = get_label_dict(zip_path=zip_path, prefix="", postfix=".png")
    train_dataset = MelDataset(
        target_dir=train_dir, labels_dict=label_dict, transform=image_transform
    )

    # size dimension is not how it is expected
    if n_images:
        train_dataset = Subset(train_dataset, torch.arange(0, n_images))
        logger.info("Number images to process %d", n_images)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=24, shuffle=True, num_workers=0
    )

    lr = params["train"]["lr"]
    epochs = params["train"]["epochs"]

    model = get_torch_model(params=params, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("TRAIN TO PREDICT STARTS ...")
    results = train_to_predict(
        model=model,
        train_dataloader=train_dataloader,
        epochs=epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    save_results_as_csv(results=results, file_path=train_predict_path_csv)
    make_folder(model_path.parent)
    save_data_as_pkl(model, model_path)
    logger.info("TRAIN FINISHED")


if __name__ == "__main__":
    train_to_predict_stage()
