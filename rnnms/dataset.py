from pathlib import Path
import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class VocoderDataset(Dataset):
    def __init__(self, root, sample_frames=24, hop_length=200):
        """
        Args:
            sample_frames: Clipped Mel-spectrogram length
            hop_length: STFT stride
        """

        self.root = Path(root)
        self.sample_frames = sample_frames
        self.hop_length = hop_length

        # Extract data pathes from meta file
        metadata_path = self.root / "train.json"
        with open(metadata_path) as file:
            metadata = json.load(file)
            self.metadata = [Path(path) for _, path in metadata]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        path = self.root / self.metadata[index]

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

        start = random.randint(0, mel.shape[-1] - self.sample_frames - 1)

        # Mel-spectrogram clipping
        start_mel = start
        end_mel = start + self.sample_frames
        # (T, freq) -> (sample_frames, freq)
        mel = mel[:, start_mel : start + end_mel]

        # Waveform clipping
        start_wave = self.hop_length * start
        end_wave = self.hop_length * (start + self.sample_frames) + 1
        # (S,) -> (samle_frames * hop_length,)
        audio = audio[start_wave : end_wave]

        return torch.LongTensor(audio), torch.FloatTensor(mel.T)