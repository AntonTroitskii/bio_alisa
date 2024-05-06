import io

import librosa
import numpy as np
import soundfile
from matplotlib import pyplot as plt
from PIL import Image


def get_mel_db(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def get_im(byte_file):
    y, sr = soundfile.read(byte_file)
    mel_db = get_mel_db(y=y, sr=sr)
    librosa.display.specshow(mel_db, sr=sr)
    # clear img_buf and save data
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    # clear matplotlib objects
    plt.close("all")
    im = Image.open(img_buf)
    return img_buf, im


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


# https://stackoverflow.com/questions/56719138/how-can-i-save-a-librosa-spectrogram-plot-as-a-specific-sized-image/57204349#57204349
def save_im_02(byte_file, output_path):
    # get mel-spectrogramm
    y, sr = soundfile.read(byte_file)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # min-max scale to fit inside 8-bit range
    ar = scale_minmax(mel_db, 0, 255).astype(np.uint8)
    ar = np.flip(ar, axis=0)  # put low frequencies at the bottom in image
    ar = 255 - ar  # invert. make black==more energy

    # save as PNG
    img = Image.fromarray(ar)
    img.save(output_path)
    img.close()

    return ar


# previous version with colr image
def save_im_01(byte_file, output_path):
    y, sr = soundfile.read(byte_file)
    mel_db = get_mel_db(y=y, sr=sr)
    librosa.display.specshow(mel_db, sr=sr)
    # clear img_buf and save data
    plt.savefig(output_path, format="png")
    # clear matplotlib objects
    plt.close("all")
