from pathlib import Path
from zipfile import ZipFile


def extract_zip_in_foler(zip_path: Path, folder_path: Path):
    with ZipFile(zip_path) as zip_file:
        zip_file.extractall(path=folder_path)


def extract_file_from_zip(zip_path: Path, file_name: str,  output_dir: Path, ):
    with ZipFile(zip_path, 'r') as zip_file:
        print(file_name)
        zip_info = zip_file.getinfo(file_name)
        zip_info.filename = Path(zip_info.filename).name
        zip_file.extract(member=zip_info, path=output_dir)
