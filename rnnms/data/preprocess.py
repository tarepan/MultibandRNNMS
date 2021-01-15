from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pyloudnorm as pyln
from torch import FloatTensor, LongTensor, save


def melspectrogram(wave: np.ndarray, sr: int, hop_length: int, win_length: int) -> np.ndarray:
    """wave2mel preprocessing.

    wave => preemphasised wave => mel => logmel => normalization
    mel - lower-cut (50Hz) log-mel amplitude spectrogram
    n_fft (2048) >> win_length (800), so information is came from only center of bin (1/4 overlap).

    Args:
        wave: waveform
        sr: sampling rate of `wav`
        hop_length: STFT stride
        win_length: STFT window length
    """

    # Hardcoded hyperparams.
    n_fft = 2048
    preemph = 0.97
    top_db = 80
    ref_db = 20
    # from paper, 'with 80 coefficients and frequencies ranging from 50 Hz to 12 kHz.' (12 kHz = sr/2)
    n_mels = 80
    fmin = 50

    mel = librosa.feature.melspectrogram(
        librosa.effects.preemphasis(wave, coef=preemph),
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin, # fmax is default sr/2
        norm=1,
        power=1, # amplitude/energy
    )
    # [-60dB, 20dB, +inf) -> (linear) -> [-1, 0, +inf)
    logmel = librosa.amplitude_to_db(mel, top_db=None) - ref_db # relative to 20dB
    logmel = np.maximum(logmel, -top_db) # clip with lowest relative -80dB
    return logmel / top_db # range==[-1, +inf]


def mu_compress(wave: np.ndarray, bits_mu_law: int, stft_hop_length: int, stft_win_length: int) -> np.ndarray:
    """Waveform μ-law compression.

    m bits waveform => bits_mu_law bits μ-law encoded waveform.

    Args:
        wave: Target waveform
        bits_mu_law: μ-law compressed waveform's bit depth
        stft_hop_length: STFT stride
        stft_win_length: STFT window length
    """

    # Pad both side of waveform. Pad length is full cover of STFT (stft_win_length//2).
    wave = np.pad(wave, (stft_win_length // 2,), mode="reflect")
    # Clip for Mel-wave shape match
    wave = wave[: ((wave.shape[0] - stft_win_length) // stft_hop_length + 1) * stft_hop_length]
    wave = 2 ** (bits_mu_law - 1) + librosa.mu_compress(wave, mu=2 ** bits_mu_law - 1)
    return wave


def preprocess_mel_mulaw(
    path_i_wav: Path,
    path_o_mel: Path,
    path_o_mulaw: Path,
    new_sr: Optional[int],
    stft_hop_length: int,
) -> None:
    """Transform LJSpeech corpus contents into mel-spectrogram and μ-law waveform.

    Before this preprocessing, corpus contents should be deployed.
    wave: Loudness norm + μ-law compression
    spec: wave Loudness norm + `melspectrogram`

    Args:
        wav_path: Path of target waveform
        id: Identity of the waveform
        dir_dataset: Path of dataset directory
        new_sr: Resample-target sampling rate
    """

    # Hardcoded hyperparams.
    win_length = 800
    bits_mulaw = 10

    # Load wav
    wave: np.ndarray
    sr: int
    wave, sr = librosa.load(path_i_wav, sr=new_sr)

    # Loudness normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wave)
    wave = pyln.normalize.loudness(wave, loudness, -24)
    peak = np.abs(wave).max()
    if peak >= 1:
        wave = wave / peak * 0.999

    # wave -> mel
    logmel = melspectrogram(wave, sr, stft_hop_length, win_length)

    # wave -> μ-law
    mulaw = mu_compress(wave, bits_mulaw, stft_hop_length, win_length)

    # save
    path_o_mel.parent.mkdir(parents=True, exist_ok=True)
    save(FloatTensor(logmel.T), path_o_mel)
    path_o_mulaw.parent.mkdir(parents=True, exist_ok=True)
    save(LongTensor(mulaw), path_o_mulaw)