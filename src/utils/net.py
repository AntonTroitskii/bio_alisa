import requests
from urllib.parse import urlencode
import tqdm
import shutil
import os
from pathlib import Path

# https://www.alpharithms.com/progress-bars-for-python-downloads-580122/


def download_file(download_url, file_path):

    # # Загружаем файл и сохраняем ег
    # download_response = requests.get(download_url, stream=True)

    # make an HTTP request within a context manager
    with requests.get(download_url, stream=True) as r:

        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))

        # implement progress bar via tqdm
        with tqdm.tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:

            # save the output to a file
            # with open(f"{os.path.basename(r.url)}", 'wb')as output:
            file_path = Path(file_path)
            with open(file_path, 'wb') as output:
                shutil.copyfileobj(raw, output)


# https://ru.stackoverflow.com/questions/1088300/%D0%BA%D0%B0%D0%BA-%D1%81%D0%BA%D0%B0%D1%87%D0%B8%D0%B2%D0%B0%D1%82%D1%8C-%D1%84%D0%B0%D0%B9%D0%BB%D1%8B-%D1%81-%D1%8F%D0%BD%D0%B4%D0%B5%D0%BA%D1%81-%D0%B4%D0%B8%D1%81%D0%BA%D0%B0
def download_yandex_file(public_key, file_path):

    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = public_key  # Сюда вписываете вашу ссылку

    # Получаем загрузочную ссылку
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_file(download_url=download_url, file_path=file_path)
