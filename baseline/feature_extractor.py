import config as cfg
import os
import time
from utils.Logger import LOG
import pandas as pd
from utils.utils import read_audio
import librosa
import numpy as np


def calculate_other_features(audio):
    """
    Calculate  spectral features other than melspectrogram from raw audio waveform
    Note: The parameters of the spectrograms are in the config.py file.
    Args:
        audio : numpy.array, raw waveform to compute the spectrogram

    Returns:
        numpy.array
        containing the spectral flatness
    """
    # Compute spectrogram
    ham_win = np.hamming(cfg.n_window)

    spec = librosa.stft(
        audio,
        n_fft=cfg.n_window,
        hop_length=cfg.hop_length,
        window=ham_win,
        center=True,
        pad_mode='reflect'
    )
    flatness = librosa.feature.spectral_flatness(
        S=np.abs(spec)
    )
    flatness_array = librosa.amplitude_to_db(flatness)
    flatness = np.mean(flatness_array)
    return flatness


def get_audio_dir_path_from_meta(filepath):
    """ Get the corresponding audio dir from a meta filepath

    Args:
        filepath : str, path of the meta filename (csv)

    Returns:
        str
        path of the audio directory.
    """
    base_filepath = os.path.splitext(filepath)[0]
    audio_dir = base_filepath.replace("metadata", "audio")
    if audio_dir.split(os.path.sep)[-2] == 'validation':
        audio_dir = (os.path.sep).join(audio_dir.split(os.path.sep)[:-1])
    audio_dir = os.path.abspath(audio_dir)
    return audio_dir


def extract_features_from_meta(csv_audio):
    """Extract features other than melspectrogram

    Args:
        csv_audio : str, file containing names, durations and labels : (name, start, end, label, label_index)
            the associated wav_filename is Yname_start_end.wav
    """
    t1 = time.time()
    df_meta = pd.read_csv(csv_audio, header=0, sep="\t")
    csv_filename = csv_audio.rsplit('\\', 1)[-1]
    out_path = os.path.join(cfg.workspace, cfg.features, csv_filename)
    LOG.info("Printing features to {} ".format(out_path))
    flatness_list = []
    for ind, wav_name in enumerate(df_meta.filename):
        if ind % 500 == 0:
            LOG.debug(ind)
        wav_dir = get_audio_dir_path_from_meta(csv_audio)
        wav_path = os.path.join(wav_dir, wav_name)

        if not os.path.isfile(wav_path):
            # LOG.error("File %s is in the csv file but the feature is not extracted!" % wav_path)
            df_meta = df_meta.drop(df_meta[df_meta.filename == wav_name].index)
        else:
            (audio, _) = read_audio(wav_path, cfg.sample_rate)
            if audio.shape[0] == 0:
                print("File %s is corrupted!" % wav_path)
            else:
                flatness_list.append(calculate_other_features(audio))
    LOG.info("Done")
    df_meta["Spectral flatness"] = flatness_list
    df_meta.to_csv(out_path, sep='\t')
    return df_meta.reset_index(drop=True)


if __name__ == '__main__':

    # extract_features_from_meta(os.path.join(cfg.workspace, cfg.validation))
    # extract_features_from_meta(os.path.join(cfg.workspace, cfg.unlabel))
    extract_features_from_meta(os.path.join(cfg.workspace, cfg.synthetic))
    extract_features_from_meta(os.path.join(cfg.workspace, cfg.weak))



