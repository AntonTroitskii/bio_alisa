from pathlib import Path
import zipfile
import librosa
import librosa.display
import io
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import yaml
from multiprocessing import Pool
from functools import partial

from zipfile import ZipFile, ZipExtFile, ZipInfo
from src.utils.files import make_folder
from src.utils.logs import get_logger
import logging


def y_sr_process_save(file_output, y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig = librosa.display.specshow(mel_db)
    plt.savefig(file_output)
    # plt.close()
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')


def process_file(file_path, file_output):
    y, sr = librosa.load(file_path)
    y_sr_process_save(file_output, y, sr)


def process_save_file(file_path, output_dir, files_set):
    if '.wav' in file_path.name:
        file_output = output_dir / (file_path.stem + '.jpg')
        if not file_output in files_set:
            process_file(file_path=file_path, file_output=file_output)


def process_files_in_dir(input_dir: Path, output_path: Path,
                         files_set, ls=500, pool_num=10):
    lf = list(input_dir.iterdir())  # list of files
    for x in range(0, len(lf), ls):
        p = Pool(pool_num)
        p.map(partial(process_save_file, output_dir=output_path,
              files_set=files_set), lf[x:x+ls])
        print(x, x+ls)


def process_byte_file(file_path, file_output):
    with io.BytesIO(file_path.read()) as byte_stream:
        y, sr = soundfile.read(byte_stream)
        y_sr_process_save(file_output, y, sr)


def get_output_path(file_name: str, output_dir):
    return output_dir / Path(file_name).with_suffix('.jpg')


def process_zip_batch(files: list[str], zip_path: Path, output_dir: Path):
    with ZipFile(zip_path) as zip_file:
        for file in files:
            with zip_file.open(file) as wav_file:
                file_name = Path(file).stem
                file_path = get_output_path(file_name, output_dir)
                process_byte_file(wav_file, file_path)


def process_zip_file(zip_path: Path, output_dir: Path, logger: logging.Logger, ls=500, pool_num=10):
    with ZipFile(zip_path) as zip_file:
        files = zip_file.namelist()

    wav_files = [file_name for file_name in files if '.wav' in file_name]
    logger.info('There are {} file(s)'.format(len(wav_files)))

    ls = len(wav_files) // pool_num
    if ls > 50:
        ls = 20

    # get batch of wav files
    files_batch = [wav_files[x: x+ls]
                   for x in range(0, len(wav_files), ls)]
    logger.info('There are {} batch(es) with {} file(s) in each except the last where {} file(s).'.format(
        len(files_batch), ls, len(files_batch[-1])))

    p = Pool(pool_num)
    count_files = 0
    for bi in range(0, len(files_batch), pool_num):
        p.map(partial(process_zip_batch, zip_path=zip_path,
                      output_dir=output_dir), files_batch[bi: bi+pool_num])

        count_files += sum([len(batch)
                           for batch in files_batch[bi: bi+pool_num]])
        logger.info('{} file(s) processed'.format(count_files))


def process_data_files():

    with open('params.yaml') as file:
        params = yaml.safe_load(file)

    logger = get_logger("DATA PROCESS", log_level=params['base']['log_level'])

    train_zip = Path(params['data']['train_zip'])
    test_zip = Path(params['data']['test_zip'])

    train_mels_dir = Path(params['data']['train_mels'])
    test_mels_dir = Path(params['data']['test_mels'])

    logger.info('Create train mels directory.')
    make_folder(train_mels_dir)
    logger.info('Start process of {}'.format(train_zip))
    process_zip_file(train_zip, train_mels_dir, logger=logger)

    logger.info('Create test mels directory.')
    make_folder(test_mels_dir)
    logger.info('Start process of {}'.format(test_zip))
    process_zip_file(test_zip, test_mels_dir, logger=logger)

    # DEBUG
    # zip_path = Path('./data/train_test.zip')
    # zip_path = Path('./data/zip/test.zip')
    # output_path = Path('./data/tes`t_output')
    # make_folder(output_path)
    # process_zip_file(zip_path, output_path)


if __name__ == '__main__':
    process_data_files()
