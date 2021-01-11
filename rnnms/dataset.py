import numpy as np
import torch
import os
from random import randint
from torch.utils.data import Dataset


class VocoderDataset(Dataset):
    def __init__(self, meta_file: str, sample_frames: int, audio_slice_frames: int, hop_length: int):
        """
        Args:
            meta_file: data path info
            sample_frames: Clipped Mel-spectrogram length
            audio_slice_frames: Clipped audio length in the unit of Mel-spectrogram (`3` == `3*hop_length` audio seq)
            hop_length: STFT stride
        """

        self.sample_frames = sample_frames
        self.audio_slice_frames = audio_slice_frames
        self.pad = (sample_frames - audio_slice_frames) // 2
        self.hop_length = hop_length

        # Extract data pathes from meta file
         with open(meta_file, encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f]
        self.metadata = [m for m in self.metadata if int(m[3]) > 2 * sample_frames - audio_slice_frames]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        _, audio_path, mel_path, _ = self.metadata[index]

        audio = np.load(os.path.join(audio_path))
        mel = np.load(os.path.join(mel_path))

        # fixed length clipping
        # pos: start
        pos = randint(0, len(mel) - self.sample_frames)

        # (T, freq) -> (sample_frames, freq)
        mel = mel[pos:pos + self.sample_frames, :]

        # (S,) -> (hop_length * audio_slice_frames,)
        p_audio = self.hop_length * (pos + self.pad)
        q_audio = self.hop_length * (pos + self.pad + self.audio_slice_frames) + 1
        audio = audio[p_audio:q_audio]

        return torch.LongTensor(audio), torch.FloatTensor(mel)
