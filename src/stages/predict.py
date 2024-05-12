from pathlib import Path

import torch
from PIL import Image

from mllib.utils.dvc import load_params
from mllib.utils.logs import get_logger
from mllib.utils.pkl import load_data_from_pkl
from src.train.dataset import get_grscl_transfrom


def predict():

    logger = get_logger(name="PREDICT")
    params = load_params()
    model = load_data_from_pkl(Path(params["train"]["model"]))
    test_dir = Path(params["data"]["processed_test_dir"])
    predict_path = Path(params["results"]["predictions"])

    im_transform = get_grscl_transfrom()
    output = []
    logger.info("Predict starts ...")
    # for num, item in enumerate(test_dir.iterdir()):
    for item in test_dir.iterdir():
        name = Path(item).stem
        im = Image.open(item)
        tr_im = im_transform(im).unsqueeze(dim=0)

        model.eval()
        with torch.inference_mode():
            tr_im_pred = model(tr_im)
        prd_label = int(torch.softmax(tr_im_pred, dim=1).argmax(dim=1))

        output_line = name + "\\" + str(prd_label)
        output.append(output_line)
        # DEBUG
        # if num == 2:
        #     break

    output = "\n".join(output)
    # name, prd_label
    with open(predict_path, "w") as file:
        file.writelines(output)
    logger.info("Predict ends ...")


if __name__ == "__main__":
    predict()
