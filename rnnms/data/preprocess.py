from pathlib import Path
from dataclasses import dataclass

import librosa
import numpy as np
import pyloudnorm as pyln
from torch import FloatTensor, LongTensor, save
from omegaconf import MISSING

@dataclass
class ConfMelspectrogram:
    """Configuration of melspectrogram preprocessing.

    Args:
        sr: Sampling rate of input waveform
        n_fft: STFT sample length
        hop_length: STFT hop length
        win_length: STFT window length
        preemph:
        top_db:
        ref_db:
        n_mels: Domension of mel frequency (from paper, 'with 80 coefficients and frequencies ranging from 50 Hz to 12 kHz.' (12 kHz = sr/2))
        fmin:
    """
    sr: int = MISSING
    n_fft: int = MISSING
    hop_length: int = MISSING
    win_length: int = MISSING
    preemph: float = MISSING
    top_db: float = MISSING
    ref_db: float = MISSING
    n_mels: int = MISSING
    fmin: int = MISSING

def melspectrogram(wave: np.ndarray, conf: ConfMelspectrogram) -> np.ndarray:
    """wave2mel preprocessing.

    wave => preemphasised wave => mel => logmel => normalization
    mel: lower-cut log-mel amplitude spectrogram
    n_fft (2048) >> win_length (800), so information is came from only center of bin (1/4 overlap).

    Args:
        wave: waveform
        conf: Configuration of this processing
    """
    mel = librosa.feature.melspectrogram(
        librosa.effects.preemphasis(wave, coef=conf.preemph),
        sr=conf.sr,
        n_fft=conf.n_fft,
        hop_length=conf.hop_length,
        win_length=conf.win_length,
        n_mels=conf.n_mels,
        fmin=conf.fmin, # fmax is default sr/2
        norm=1,
        power=1, # amplitude/energy
    )
    # [-60dB, 20dB, +inf) -> (linear) -> [-1, 0, +inf)
    logmel = librosa.amplitude_to_db(mel, top_db=None) - conf.ref_db # relative to 20dB
    logmel = np.maximum(logmel, -1.0*conf.top_db) # clip with lowest relative -80dB
    return logmel / conf.top_db # range==[-1, +inf]


def fit_length(wave: np.ndarray, stft_hop_length: int, stft_win_length: int) -> np.ndarray:
    """Fit waveform length to spectrogram.

    Args:
        wave: Target waveform
        stft_hop_length: STFT stride
        stft_win_length: STFT window length
    Returns:
        Length-fit waveform
    """
    # Pad both side of waveform. Pad length is full cover of STFT (stft_win_length//2).
    wave = np.pad(wave, (stft_win_length // 2,), mode="reflect")

    # Clip for Mel-wave shape match
    wave = wave[: ((wave.shape[0] - stft_win_length) // stft_hop_length + 1) * stft_hop_length]

    return wave


def mu_compress(wave: np.ndarray, bits_mu_law: int) -> np.ndarray:
    """Waveform μ-law compression.

    m bits waveform => bits_mu_law bits μ-law encoded waveform.

    Args:
        wave: Target waveform
        bits_mu_law: Bit depth of μ-law compressed waveform
    Returns:
        μ-law encoded waveform, each sample point is int and in range [0, 2^bit - 1]
    """

    # mu-law conversion
    mu_law_librosa: np.ndarray = librosa.mu_compress(wave, mu=2 ** bits_mu_law - 1)

    # Range adaption from librosa to Categorical : [-2^(bit-1), 2^(bit-1)-1] -> [0, 2^bit - 1]
    mu_law = 2 ** (bits_mu_law - 1) + mu_law_librosa

    return mu_law


@dataclass
class ConfPreprocessing:
    """Configuration of preprocessing.

    Args:
    target_sr: Desired sampling rate of waveform
    stft_hop_length: Hop length of STFT for mel-spectrogram
    win_length: Window length of STFT for mel-spectrogram
    bits_mulaw: Bit depth of μ-law compressed wavefrom
    """
    target_sr: int = MISSING
    stft_hop_length: int = MISSING
    win_length: int = MISSING
    bits_mulaw: int = MISSING
    melspec: ConfMelspectrogram = ConfMelspectrogram(
        sr="${..target_sr}",
        hop_length="${..stft_hop_length}",
        win_length="{..win_length}"
    )

def preprocess_mel_mulaw(
    path_i_wav: Path,
    path_o_mel: Path,
    path_o_mulaw: Path,
    conf: ConfPreprocessing,
) -> None:
    """Transform LJSpeech corpus contents into mel-spectrogram and μ-law waveform.

    Before this preprocessing, corpus contents should be deployed.
    wave: Loudness norm + μ-law compression
    spec: wave Loudness norm + `melspectrogram`

    Args:
        path_i_wav: Path of target waveform
        id: Identity of the waveform
        dir_dataset: Path of dataset directory
    """

    # Load wav
    wave: np.ndarray
    sr: int
    wave, sr = librosa.load(path_i_wav, sr=conf.target_sr)

    # Loudness normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wave)
    wave = pyln.normalize.loudness(wave, loudness, -24)
    peak = np.abs(wave).max()
    if peak >= 1:
        wave = wave / peak * 0.999

    # wave -> mel
    logmel = melspectrogram(wave, conf.melspec)

    # wave -> length-adjusted wave
    wave_length_fit = fit_length(wave, conf.stft_hop_length, conf.win_length)

    # linear wave -> μ-law wave (length fit + μ-law conversion)
    mulaw = mu_compress(wave_length_fit, conf.bits_mulaw)

    # save
    path_o_mel.parent.mkdir(parents=True, exist_ok=True)
    save(FloatTensor(logmel.T), path_o_mel)
    path_o_mulaw.parent.mkdir(parents=True, exist_ok=True)
    # todo: can be `torch.ShortTensor`?
    save(LongTensor(mulaw), path_o_mulaw)
