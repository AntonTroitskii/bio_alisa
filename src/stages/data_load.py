from mllib.utils.net import download_yandex_file
from mllib.utils.dvc import load_params
from mllib.utils.io import make_folder
from pathlib import Path
# import asyncio


def load_zip_data(params):
    """
    Downloading three large files simultaneously.
    Each file has its own progress bar.
    """
    # loop = asyncio.get_running_loop()
    make_folder(Path((params['data']['zip_folder'])))
    urls = [
        (params['data']['train_link'], params['data']['train_zip']),
        (params['data']['test_link'], params['data']['test_zip'])
    ]
    # tasks = [loop.create_task(download_yandex_file(url, file_path))
    #          for url, file_path in urls]
    # await asyncio.gather(*tasks, return_exceptions=True)

    for url, path in urls:
        download_yandex_file(url, path)


def extract_zip(params):
    make_folder(Path(params['data']['unzip_dir']))
    # extract_zip_in_foler(params['data']['train_zip'],
    #                      params['data']['unzip_dir'])
    # extract_zip_in_foler(params['data']['test_zip'],
    #                      params['data']['unzip_dir'])


if __name__ == '__main__':
    params = load_params()
    load_zip_data(params)
