
from pathlib import Path
from zipfile import ZipFile
import shutil
import yaml


def clear_all(dir_path: Path):
    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        elif item.is_file():
            item.unlink()


def load_params(path='params.yaml'):
    with open(Path(path)) as config_file:
        return yaml.safe_load(config_file)


def delete_folder(path: Path):
    if path.exists() and path.is_dir():
        try:
            shutil.rmtree(path)
        except Exception as ex:
            print(ex)


def delete_files_in_folder(path: Path):
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()


def make_folder(fpath: Path):
    if fpath.exists() and fpath.is_dir():
        clear_all(fpath)
    else:
        fpath.mkdir(parents=True, exist_ok=True)


def extract_zip_in_foler(zip_path: Path, folder_path: Path):
    with ZipFile(zip_path) as zip_file:
        zip_file.extractall(path=folder_path)


def extract_file_from_zip(zip_path: Path, file_name: str,  output_dir: Path, ):
    with ZipFile(zip_path, 'r') as zip_file:
        print(file_name)
        zip_info = zip_file.getinfo(file_name)
        zip_info.filename = Path(zip_info.filename).name
        zip_file.extract(member=zip_info, path=output_dir)
