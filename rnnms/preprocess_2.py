from pathlib import Path
import hydra
import hydra.utils as utils

import json
import librosa
import numpy as np
import pyloudnorm as pyln
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def melspectrogram(
    wav,
    sr=16000,
    hop_length: int =200,
    win_length=800,
    n_fft=2048,
    n_mels=128,
    fmin=50,
    preemph=0.97,
    top_db=80,
    ref_db=20,
):
    """wave2mel preprocessing.

    wave => preemphasised wave => mel => logmel => normalization
    mel - lower-cut (50Hz) log-mel amplitude spectrogram

    n_fft (2048) >> win_length (800), so information is came from only center of bin (1/4 overlap).

    Args:
        wav: waveform
        sr: sampling rate of `wav`
        hop_length: STFT stride

        fmin: Mel-spectrogram's lowest frequency
    """
    mel = librosa.feature.melspectrogram(
        librosa.effects.preemphasis(wav, coef=preemph),
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        norm=1,
        power=1, # amplitude/energy
    )
    # [-60dB, 20dB, +inf) -> (linear) -> [-1, 0, +inf)
    logmel = librosa.amplitude_to_db(mel, top_db=None) - ref_db # relative to 20dB
    logmel = np.maximum(logmel, -top_db) # clip with lowest relative -80dB
    return logmel / top_db # range==[-1, +inf]


def mu_compress(wav, hop_length=200, frame_length=800, bits_mu_law=8):
    """μ-law compression with librosa.

    m bits waveform => bits_mu_law bits μ-law encoded waveform.     

    Args:
        hop_length: STFT stride

        ((wav.shape[0] - frame_length) // hop_length) * hop_length + hop_length 
        frame_length...?
    """
    wav = np.pad(wav, (frame_length // 2,), mode="reflect")
    wav = wav[: ((wav.shape[0] - frame_length) // hop_length + 1) * hop_length]
    wav = 2 ** (bits_mu_law - 1) + librosa.mu_compress(wav, mu=2 ** bits_mu_law - 1)
    return wav


def process_wav(wav_path, out_path, cfg):
    """Preprocessing

    wave: Loudness norm + μ-law compression
    spec: wave Loudness norm + "melspectrogram"
    """

    meter = pyln.Meter(cfg.sr)

    # Load wav
    wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=cfg.sr)

    # Loudness normalization
    loudness = meter.integrated_loudness(wav)
    wav = pyln.normalize.loudness(wav, loudness, -24)
    peak = np.abs(wav).max()
    if peak >= 1:
        wav = wav / peak * 0.999

    # wave 2 mel
    logmel = melspectrogram(
        wav,
        sr=cfg.sr,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        preemph=cfg.preemph,
        top_db=cfg.top_db,
    )

    # wave 2 μ-law
    wav = mu_compress(
        wav,
        hop_length=cfg.hop_length,
        frame_length=cfg.win_length,
        bits_mu_law=cfg.mulaw.bits,
    )

    np.save(out_path.with_suffix(".mel.npy"), logmel)
    np.save(out_path.with_suffix(".wav.npy"), wav)
    return out_path, logmel.shape[-1]


@hydra.main(config_path="univoc/config", config_name="preprocess")
def preprocess_dataset(cfg):
    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    print("Extracting features for train set")
    futures = []
    split_path = out_dir / "train"
    with open(split_path.with_suffix(".json")) as file:
        metadata = json.load(file)
        for in_path, out_path in metadata:
            wav_path = in_dir / in_path
            out_path = out_dir / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            futures.append(
                executor.submit(process_wav, wav_path, out_path, cfg.preprocess)
            )

    results = [future.result() for future in tqdm(futures)]

    lengths = {result[0].stem: result[1] for result in results}
    with open(out_dir / "lengths.json", "w") as file:
        json.dump(lengths, file, indent=4)

    frames = sum(lengths.values())
    frame_shift_ms = cfg.preprocess.hop_length / cfg.preprocess.sr
    hours = frames * frame_shift_ms / 3600
    print(f"Wrote {len(lengths)} utterances, {frames} frames ({hours:.2f} hours)")


if __name__ == "__main__":
    preprocess_dataset()