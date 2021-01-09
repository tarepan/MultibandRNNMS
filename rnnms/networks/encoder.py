from typing import Tuple
import torch
from torch.tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchaudio.transforms import MuLawDecoding
import pytorch_lightning as pl


class FakeGRU0(nn.Module):
    """Stub of GRU which return all-0 output
    """
    def __init__(self, mel_channels, conditioning_channels, device, bidi=True):
        super().__init__()
        self.input_dim = mel_channels
        self.hidden_dim = conditioning_channels
        self.device = device
        self.factor = 1
        if bidi == True:
            self.factor = 2
    
    def forward(self, mels):
        """
        Arguments:
            mels {Tensor(Batch, Time, Freq)}
        Return:
            Tensor(B, length, hidden*2)
        """
        s = mels.size()
        return torch.zeros(s[0], s[1], self.hidden_dim*self.factor).to(self.device), 0


class SpecEncoder(nn.Module):
    """2-layer bidirectional GRU mel-spectrogram encoder
    """
    def __init__(self, dim_mel_freq: int, dim_latent: int, hop_length):
        """Initiate SpecEncoder.

        Args:
            dim_mel_freq : input vector dimension == input spectrogram frequency dimension
            dim_latent : hidden/output vector dimension
            hop_length ([type]): [description]
        """
        super().__init__()
        self.hop_length = hop_length
        ## 2-layer bidirectional GRU
        self.rnn = nn.GRU(dim_mel_freq, dim_latent, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x, mels):
        """forward computation for training.
        
        Mel-Spec => latent series.
        This is RNN, so time dimension is preserved.

        Args:
            mels {Tensor(Batch, Time, Freq)} -- preprocessed mel-spectrogram
        Returns:
            Tensor(Batch, Time, 2 * dim_latent) -- time-series output latents
        """
        dim_mels_time = mels.size(1)
        audio_slice_frames = x.size(1) // self.hop_length
        pad = (dim_mels_time - audio_slice_frames) // 2

        mels, _ = self.rnn(mels)
        mels = mels[:, pad:pad + audio_slice_frames, :]
        return mels

    def generate(self, mels):
        mels, _ = self.rnn(mels)
        return mels

