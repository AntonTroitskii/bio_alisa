import zipfile
from pathlib import Path

from mllib.utils.dvc import load_params
from mllib.utils.io import make_folder
from mllib.utils.logs import get_logger


def extract_zip_to_target(zip_path, check_dir, target_dir):
    if not check_dir.exists():
        make_folder(check_dir)
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(path=target_dir)


def extract():
    params = load_params()
    processed_dir = Path(params["data"]["processed_dir"])
    processed_train_dir = Path(params["data"]["processed_train_dir"])
    processed_test_dir = Path(params["data"]["processed_test_dir"])
    processed_train_zip = Path(params["data"]["processed_train_zip"])
    processed_test_zip = Path(params["data"]["processed_test_zip"])
    logger = get_logger("DATA UNPACK", log_level=params["base"]["log_level"])
    logger.info("Start extract files")
    extract_zip_to_target(processed_train_zip, processed_train_dir, processed_dir)
    extract_zip_to_target(processed_test_zip, processed_test_dir, processed_dir)
    logger.info("Start extact files compleated.")


if __name__ == "__main__":
    extract()
