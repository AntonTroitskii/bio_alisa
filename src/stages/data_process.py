from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import yaml

from mllib.utils.io import make_folder
from mllib.utils.logs import get_logger
from src.utils.files import (
    get_files_batch,
    get_files_list,
    process_zip_batch_files,
    zip_folder,
)


def porcess_files_in_zip(zip_path: Path, output_path: Path, num_pool: int):
    wav_files = get_files_list(zip_path=zip_path, postfix=".wav")
    ls = int(len(wav_files) / num_pool)
    batches = get_files_batch(wav_files, ls)

    with ProcessPoolExecutor(max_workers=num_pool) as executor:
        for bi in range(0, len(batches), num_pool):
            executor.map(
                partial(
                    process_zip_batch_files,
                    zip_path=zip_path,
                    output_path=output_path,
                    postfix=".png",
                ),
                batches[bi : bi + num_pool],
            )


def process_data_files():

    with open("params.yaml") as file:
        params = yaml.safe_load(file)

    num_pool = params["base"]["num_pool"]

    train_zip = Path(params["data"]["train_zip"])
    test_zip = Path(params["data"]["test_zip"])
    processed_dir = Path(params["data"]["processed_dir"])
    processed_train_dir = Path(params["data"]["processed_train_dir"])
    processed_test_dir = Path(params["data"]["processed_test_dir"])

    processed_train_zip = Path(params["data"]["processed_train_zip"])
    processed_test_zip = Path(params["data"]["processed_test_zip"])

    logger = get_logger("DATA PROCESS", log_level=params["base"]["log_level"])

    logger.info("Start extract files")
    make_folder(processed_dir)

    make_folder(processed_train_dir)
    porcess_files_in_zip(
        zip_path=train_zip, output_path=processed_train_dir, num_pool=num_pool
    )
    zip_folder(zip_path=processed_train_zip, input_path=processed_train_dir)
    # delete_folder(processed_train_dir)

    make_folder(processed_test_dir)
    porcess_files_in_zip(
        zip_path=test_zip, output_path=processed_test_dir, num_pool=num_pool
    )
    zip_folder(zip_path=processed_test_zip, input_path=processed_test_dir)
    # delete_folder(processed_test_dir)


if __name__ == "__main__":
    process_data_files()


# def y_sr_process_save(file_output, y, sr):
#     mel = librosa.feature.melspectrogram(y=y, sr=sr)
#     mel_db = librosa.power_to_db(mel, ref=np.max)
#     fig = librosa.display.specshow(mel_db)
#     plt.savefig(file_output)
#     # plt.close()
#     # Clear the current axes.
#     plt.cla()
#     # Clear the current figure.
#     plt.clf()
#     # Closes all the figure windows.
#     plt.close("all")


# def process_file(file_path, file_output):
#     y, sr = librosa.load(file_path)
#     y_sr_process_save(file_output, y, sr)


# def process_save_file(file_path, output_dir, files_set):
#     if ".wav" in file_path.name:
#         file_output = output_dir / (file_path.stem + ".jpg")
#         if not file_output in files_set:
#             process_file(file_path=file_path, file_output=file_output)


# def process_files_in_dir(
#     input_dir: Path, output_path: Path, files_set, ls=500, pool_num=10
# ):
#     lf = list(input_dir.iterdir())  # list of files
#     for x in range(0, len(lf), ls):
#         p = Pool(pool_num)
#         p.map(
#             partial(process_save_file, output_dir=output_path, files_set=files_set),
#             lf[x : x + ls],
#         )


# def process_byte_file(file_path, file_output):
#     with io.BytesIO(file_path.read()) as byte_stream:
#         y, sr = soundfile.read(byte_stream)
#         y_sr_process_save(file_output, y, sr)


# def process_zip_batch(files: list[str], zip_path: Path, output_dir: Path):
#     with ZipFile(zip_path) as zip_file:
#         for file in files:
#             with zip_file.open(file) as wav_file:
#                 file_name = Path(file).stem
#                 file_path = get_output_path(file_name, output_dir)
#                 process_byte_file(wav_file, file_path)


# def get_labeld_files(zip_path, file_name, stem=".wav", parent_path="train/"):
#     files_0 = []
#     files_1 = []
#     with ZipFile(zip_path) as zip_file:
#         with zip_file.open(file_name) as file:
#             for line in file.readlines():
#                 name, label = line.split()
#                 name = name.decode("utf-8")
#                 label = int(label.decode("utf-8"))
#                 path = parent_path + name + stem

#                 if label == 0:
#                     files_0.append(path)
#                 elif label == 1:
#                     files_1.append(path)

#     return files_0, files_1


# def process_zip_batch_files(
#     zip_path: Path,
#     output_dir: Path,
#     logger: logging.Logger,
#     files_batch: list[list[str]],
#     pool_num: int,
#     ls: int,
# ):
#     # get batch of wav files
#     logger.info(
#         "There are {} batch(es) with {} file(s) in each except the last where {} file(s).".format(
#             len(files_batch), ls, len(files_batch[-1])
#         )
#     )

#     p = Pool(pool_num)
#     count_files = 0
#     for bi in range(0, len(files_batch), pool_num):
#         p.map(
#             partial(process_zip_batch, zip_path=zip_path, output_dir=output_dir),
#             files_batch[bi : bi + pool_num],
#         )

#         count_files += sum([len(batch) for batch in files_batch[bi : bi + pool_num]])
#         logger.info("{} file(s) processed".format(count_files))


# def process_zip_file(
#     zip_path: Path,
#     output_dir: Path,
#     logger: logging.Logger,
#     labels_file_name: str = "",
#     pool_num=10,
#     ls=20,
# ):
#     with ZipFile(zip_path) as zip_file:
#         files = zip_file.namelist()
#         wav_files = [file_name for file_name in files if ".wav" in file_name]
#         logger.info("There are {} file(s)".format(len(wav_files)))

#     if not labels_file_name:
#         files_batch = get_files_batch(files=wav_files, ls=ls)

#         process_zip_batch_files(
#             zip_path=zip_path,
#             output_dir=output_dir,
#             logger=logger,
#             files_batch=files_batch,
#             pool_num=pool_num,
#             ls=ls,
#         )

#     else:
#         files_0, files_1 = get_labeld_files(
#             zip_path=zip_path,
#             file_name=labels_file_name,
#             parent_path="train/",
#             stem=".wav",
#         )

#         files_batch_0 = get_files_batch(files=files_0, ls=ls)
#         output_dir_0 = output_dir / "0"
#         make_folder(output_dir_0)
#         process_zip_batch_files(
#             zip_path=zip_path,
#             output_dir=output_dir_0,
#             logger=logger,
#             files_batch=files_batch_0,
#             pool_num=pool_num,
#             ls=ls,
#         )

#         files_batch_1 = get_files_batch(files=files_1, ls=ls)
#         output_dir_1 = output_dir / "1"
#         make_folder(output_dir_1)
#         process_zip_batch_files(
#             zip_path=zip_path,
#             output_dir=output_dir_1,
#             logger=logger,
#             files_batch=files_batch_1,
#             pool_num=pool_num,
#             ls=ls,
#         )
