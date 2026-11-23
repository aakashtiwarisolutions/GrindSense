import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000
N_FFT, HOP, N_MELS = 512, 320, 64

IN_DIST = ['snoring', 'breathing', 'coughing', 'sneezing', 'clock_tick', 'door_wood_creaks', 'footsteps', 'keyboard_typing']
NOISE = ['rain', 'wind', 'washing_machine', 'vacuum_cleaner', 'engine', 'thunderstorm']
UNSEEN = ['dog', 'cat', 'crying_baby', 'clock_alarm', 'siren', 'car_horn', 'laughing', 'toilet_flush']


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=8000.0):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        for k in range(a, b):
            fb[i - 1, k] = (k - a) / max(b - a, 1)
        for k in range(b, c):
            fb[i - 1, k] = (c - k) / max(c - b, 1)
    return fb


FB = mel_filterbank()
WIN = np.hanning(N_FFT)


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(1)
    return resample_poly(x, 160, 441).astype(np.float32) if sr == 44100 else x


def logmel(x):
    pad = np.pad(x, (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(pad, N_FFT)[::HOP] * WIN
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(FB @ power.T + 1e-6).astype(np.float32)


def main(root, out):
    meta = pd.read_csv(Path(root) / 'meta' / 'esc50.csv')
    keep = meta[meta.category.isin(IN_DIST + NOISE + UNSEEN)].reset_index(drop=True)
    waves = np.stack([load(Path(root) / 'audio' / f) for f in keep.filename])
    keep.to_csv(Path(out) / 'clips.csv', index=False)
    np.save(Path(out) / 'waves.npy', waves)
    print(keep.category.value_counts().to_dict(), waves.shape)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
