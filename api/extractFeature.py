import numpy as np
import librosa

audios_esc = "/content/ESC-50/audio"
audios_audioset = "/content/audios_total"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Carregar o arquivo de áudio

    # Extrair MFCCs com coeficientes 13
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    mfccs_max = np.max(mfccs.T, axis=0)
    mfccs_min = np.min(mfccs.T, axis=0)
    mfccs_median = np.median(mfccs.T, axis=0)

    # Delta e Delta-Delta MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfccs_mean = np.mean(delta_mfccs.T, axis=0)
    delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
    delta_delta_mfccs_mean = np.mean(delta_delta_mfccs.T, axis=0)

    # Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zero_crossing_rate)

    # Short-Time Energy
    short_time_energy = np.array([np.sum(frame ** 2) for frame in
                                  librosa.util.frame(y, frame_length=2048, hop_length=512)])
    ste_mean = np.mean(short_time_energy)

    # Concatenar todas as características em um único vetor de características
    features = np.concatenate((mfccs_mean, mfccs_std, mfccs_max, mfccs_min, mfccs_median,
                               delta_mfccs_mean, delta_delta_mfccs_mean,
                               [zcr_mean, ste_mean]))

    return features