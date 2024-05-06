import os
import tarfile
import zipfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import soundfile
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from src.utils.data_process import save_im_02


def extract_zip(zip_path: Path, folder_path: Path):
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(path=folder_path)


def extract_file_from_zip(
    zip_path: Path,
    file_path: str,
    output_dir: Path,
):
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_info = zip_file.getinfo(file_path)
        zip_info.filename = Path(zip_info.filename).name
        zip_file.extract(member=zip_info, path=output_dir)


def extract_files_list(zip_path: Path, files: list, output: Path):
    with zipfile.ZipFile(zip_path) as zip_file:
        for file_path in files:
            zip_info = zip_file.getinfo(file_path)
            file_name = Path(zip_info.filename).name
            with open(output / file_name, "wb") as file:
                file.write(zip_file.read(zip_info))


def update_path(path: str, prefix, postfix):
    return prefix + path + postfix


def get_label_dict(
    zip_path,
    targets_file: str = "train/targets.tsv",
    prefix="train/",
    postfix=".wav",
):
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(targets_file) as byte_file:
            df = pd.read_csv(byte_file, names=["file_name", "label"], sep="\t")
            df["file_name"] = df["file_name"].map(lambda x: prefix + x + postfix)
            # partial(update_path, prefix=prefix, postfix=postfix)
            # )
            return df.set_index("file_name")["label"].to_dict()


def get_list_files(
    zip_path: Path,
    targets_file: str = "train/targets.tsv",
    prefix="train/",
    postfix=".wav",
):
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(targets_file) as byte_file:
            df = pd.read_csv(byte_file, names=["file_name", "label"], sep="\t")
            file_path_list = [prefix + i + postfix for i in df["file_name"].tolist()]
            return file_path_list


def split_train_val_test(files_list: list, fraction=[0.7, 0.2, 0.1], seed=42):
    generator1 = torch.Generator().manual_seed(seed)
    ind_list = [
        list(i) for i in random_split(files_list, fraction, generator=generator1)
    ]
    return ind_list


def read_y_sr_from_zip(zip_path: Path, file_name: str):
    """Read y and sr of soundfile which is in zip file;

    Args:
        zip_file (Path): path to zip file,
        file_path_zip (str): path for file in zip file.

    Returns:
        tupoe(numpy.ndarray, int): y and sr of sound file
    """
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(file_name) as byte_file:
            y, sr = soundfile.read(byte_file)

    return y, sr


def get_files_list(zip_path: Path, postfix=""):
    with ZipFile(zip_path) as zip_file:
        files = zip_file.namelist()
        wav_files = [file_name for file_name in files if postfix in file_name]
        return wav_files


def get_files_batch(files: list, batch_size: int):
    return [files[x : x + batch_size] for x in range(0, len(files), batch_size)]


def get_output_path(file_path: str, output_dir: Path, postfix=""):
    return output_dir / (Path(file_path).stem + postfix)


def process_zip_batch_files(files_batch, zip_path, output_path, postfix):
    with zipfile.ZipFile(file=zip_path) as zip_file:
        for file_path in tqdm(files_batch):
            with zip_file.open(file_path) as byte_file:
                file_path_output = get_output_path(
                    file_path=file_path, output_dir=output_path, postfix=postfix
                )
                save_im_02(byte_file=byte_file, output_path=file_path_output)


def zip_folder(zip_path, input_path):
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for folderName, subfolders, filenames in os.walk(input_path):
            for filename in tqdm(filenames):
                file_path = Path(folderName) / filename
                parent = file_path.parent.parent
                arcname = file_path.relative_to(parent)
                zip_file.write(filename=file_path, arcname=arcname)


def count_number_files_in_tar(tar_path: Path, suffix=""):
    with tarfile.open(tar_path, "r") as tar:
        return ([filename.endswith(suffix) for filename in tar.getnames()]).count(True)


def count_number_files_in_tars(folder_path: Path, suffix=""):
    s = 0
    desc = "Counting {} files in folder: {}".format(suffix, folder_path)
    for tar_path in tqdm(
        [Path(tar_path) for tar_path in folder_path.glob("*.tar")], desc=desc
    ):
        s += count_number_files_in_tar(Path(tar_path), suffix=suffix)

    return s
