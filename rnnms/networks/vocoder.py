from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MuLawDecoding
from omegaconf import MISSING

from .decoder import C_eAR_GenRNN, ConfC_eAR_GenRNN


@dataclass
class ConfRNN_MS_Vocoder:
    """Configuration of RNN_MS_Vocoder.
    Args:
    size_mel_freq: Dimension of mel frequency
    size_latent: Dimension of latent vector
    bits_mu_law: Bit depth of μ-law encoding
    hop_length: STFT stride
    """
    size_mel_freq: int = MISSING
    size_latent: int = MISSING
    bits_mu_law: int = MISSING
    hop_length: int = MISSING
    wave_ar: ConfC_eAR_GenRNN = ConfC_eAR_GenRNN(size_i_cnd="${..size_latent}", size_o_bit="${..bits_mu_law}")

class RNN_MS_Vocoder(nn.Module):
    """RNN_MS: Universal Vocoder
    """
    def __init__(self, conf: ConfRNN_MS_Vocoder):
        super().__init__()
        self.hop_length = conf.hop_length

        # PreNet which transform conditioning inputs into latent representation
        self.prenet = nn.GRU(conf.size_mel_freq, conf.size_latent, num_layers=2, batch_first=True, bidirectional=True)
        # AR which yeild sample probability autoregressively
        self.ar = C_eAR_GenRNN(conf.wave_ar)
        self.mulaw_dec = MuLawDecoding(2**conf.bits_mu_law)

    def forward(self, wave_mu_law: Tensor, mels: Tensor) -> Tensor:
        """Forward computation for training.
        
        Arguments:
            wave_mu_law: mu-law encoded waveform
            mels (Tensor(Batch, Time, Freq)): preprocessed mel-spectrogram
        
        Returns:
            Tensor(Batch, Time, before_softmax) Generated mu-law encoded waveform
        """
        # Latent representation
        # Tensor(batch, T_mel, 2*size_latent)
        latents, _ = self.prenet(mels)

        # Cond. Upsampling
        # Tensor(batch, T_mel*hop_length, 2*size_latent)
        latents_upsampled: Tensor = F.interpolate(latents.transpose(1, 2), scale_factor=self.hop_length).transpose(1, 2)

        # Autoregressive
        bits_energy_series = self.ar(wave_mu_law, latents_upsampled)

        return bits_energy_series

    def generate(self, mel: Tensor) -> Tensor:
        """Generate waveform from mel-spectrogram.

        Input has variable length, and long zero padding is not good for RNN, so batch_size must be 1.

        Args:
            mel (Tensor(1, T_mel, freq)): Mel-spectrogram tensor. 
        Returns:
            (Tensor(1, T_mel * hop_length)) Generated waveform. A sample point is in range [-1, 1].
        """

        # Transform conditionings into upsampled latents
        latents, _ = self.prenet(mel)
        latents_upsampled: Tensor = F.interpolate(latents.transpose(1, 2), scale_factor=self.hop_length).transpose(1, 2)

        # Sample a waveform (sequence of μ-law encoded samples)
        output_mu_law = self.ar.generate(latents_upsampled)

        # μ-law expansion.
        # range: [0, 2^bit -1] => [-1, 1]
        # [PyTorch](https://pytorch.org/audio/stable/transforms.html#mulawdecoding)
        # bshall/UniversalVocoding use librosa, so range adaption ([0, n] -> [-n/2, n/2]) was needed.
        output_wave: Tensor = self.mulaw_dec(output_mu_law)

        return output_wave