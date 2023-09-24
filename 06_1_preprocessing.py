from pathlib import Path
import zipfile
import librosa, librosa.display
import io
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import yaml
from  multiprocessing import Pool
from functools import partial

with open('params.yaml') as file:
    params = yaml.safe_load(file)

def save_mel_from_zip(file_path, file_output):
    y, sr = soundfile.read(io.BytesIO(file_path.read_bytes()))
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref = np.max)
    librosa.display.specshow(mel_db)
    plt.savefig(file_output)
    
def save_mel(file_path, file_output):
    y, sr = librosa.load(file_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref = np.max)
    librosa.display.specshow(mel_db)
    plt.savefig(file_output)
    
    
root = Path(params['data']['root'])
train_zip = Path(params['data']['train_zip'])
train_dir = Path(params['data']['train_dir'])
train_mels_dir = Path(params['data']['train_mels'])

test_zip = Path(params['data']['test_zip'])
test_dir = Path(params['data']['test_dir'])
test_mels_dir = Path(params['data']['test_mels'])

def save_mel(file_path, file_output):
    y, sr = librosa.load(file_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref = np.max)
    librosa.display.specshow(mel_db)
    plt.savefig(file_output)


def save_file(file_path, dir):
    if '.wav' in file_path.name:
        file_output = dir / (file_path.stem + '.jpg')
        if not file_output.exists():
            save_mel(file_path=file_path, file_output=file_output)


def process_files(input, output, ls=500, pool_num=10):
    ls = 500 # list size
    lf = list(input.iterdir()) # list of files
    for x in range(0, len(lf), ls):
        p = Pool(pool_num)
        p.map(partial(save_file, dir=output), lf[x:x+ls])
        print(x, x+ls)

if __name__ == '__main__':
    
    process_files(train_dir, train_mels_dir)
    process_files(test_dir, test_mels_dir)