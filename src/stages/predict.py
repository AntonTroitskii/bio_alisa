from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from mllib.utils.dvc import load_params
from mllib.utils.logs import get_logger
from mllib.utils.pkl import load_data_from_pkl
from src.train.dataset import get_grscl_transfrom


def predict():
    params = load_params()

    logger = get_logger(name="PREDICT")
    model_path = Path(params["predict"]["model"])
    predict_path = Path(params["results"]["predictions"])

    logger.info("PREDICT STARTS ...")
    test_dir = Path(params["data"]["processed_test_dir"])

    model = load_data_from_pkl(model_path)
    image_transform = get_grscl_transfrom()
    output_dict = {"id": [], "gender": []}

    for num, item in tqdm(
        enumerate(test_dir.iterdir()), total=len(list(test_dir.iterdir()))
    ):
        im = Image.open(item)
        tr_im = image_transform(im).unsqueeze(dim=0)

        model.eval()
        with torch.inference_mode():
            tr_im_pred = model(tr_im)
        gender = int(torch.softmax(tr_im_pred, dim=1).argmax(dim=1))
        item_id = Path(item).stem

        output_dict["id"].append(item_id)
        output_dict["gender"].append(gender)
        # DEBUG
        # if num == 2:
        #     break

    # write file
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(predict_path, index=False, sep="\t", header=False)

    logger.info("PREDICT FINISED ...")


if __name__ == "__main__":
    predict()

    # output_line = name + "\t" + str(prd_label)
    # output.append(output_line)

    # output = "\n".join(output)
    # name, prd_label
    # with open(predict_path, "w") as file:
    # file.writelines(output)
