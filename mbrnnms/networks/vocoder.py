"""Vocoder networks"""


from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MuLawDecoding
from omegaconf import MISSING

from .prenet import RecurrentPreNet, ConfRecurrentPreNet
from .decoder import C_eAR_GenRNN, ConfC_eAR_GenRNN


@dataclass
class ConfRNNMSVocoder:
    """Configuration of RNNMSVocoder.
    Args:
        dim_i_feature: Dimension of input feature (e.g. Frequency dim of mel-spec)
        dim_voc_latent: Dimension of vocoder's latent between PreNet and WaveRNN
        bits_mu_law: Bit depth of μ-law encoding
        upsampling_t: Factor of time-direcitonal latent upsampling (e.g. STFT stride)
    """
    dim_i_feature: int = MISSING
    dim_voc_latent: int = MISSING
    bits_mu_law: int = MISSING
    upsampling_t: int = MISSING
    prenet: ConfRecurrentPreNet = ConfRecurrentPreNet(
        dim_i="${..dim_i_feature}",
        dim_o="${..dim_voc_latent}")
    wave_ar: ConfC_eAR_GenRNN = ConfC_eAR_GenRNN(
        size_i_cnd="${..dim_voc_latent}",
        size_o_bit="${..bits_mu_law}")

class RNNMSVocoder(nn.Module):
    """RNN_MS Universal Vocoder, generating speech from conditioning.
    """
    def __init__(self, conf: ConfRNNMSVocoder):
        super().__init__()
        self.time_upsampling_factor = conf.upsampling_t

        # PreNet which transform conditioning inputs into latent representation
        self.prenet = RecurrentPreNet(conf.prenet)
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
        # Tensor(batch, T_mel, size_latent)
        latents = self.prenet(mels)

        # Cond. Upsampling
        # Tensor(batch, T_mel*hop_length, size_latent)
        latents_upsampled: Tensor = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=self.time_upsampling_factor
        ).transpose(1, 2)

        # Autoregressive
        bits_energy_series = self.ar(wave_mu_law, latents_upsampled)

        return bits_energy_series

    def generate(self, mel: Tensor) -> Tensor:
        """Generate waveform from mel-spectrogram.

        Input has variable length, and long zero padding is not good for RNN,
        so batch_size must be 1.

        Args:
            mel (Tensor(1, T_mel, freq)): Mel-spectrogram tensor.
        Returns:
            (Tensor(1, T_mel * hop_length)) Generated waveform. A sample point is in range [-1, 1].
        """

        # Transform conditionings into upsampled latents
        latents = self.prenet(mel)
        latents_upsampled: Tensor = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=self.time_upsampling_factor
        ).transpose(1, 2)

        # Sample a waveform (sequence of μ-law encoded samples)
        output_mu_law = self.ar.generate(latents_upsampled)

        # μ-law expansion.
        # range: [0, 2^bit -1] => [-1, 1]
        # [PyTorch](https://pytorch.org/audio/stable/transforms.html#mulawdecoding)
        # bshall/UniversalVocoding use librosa,
        # so range adaption ([0, n] -> [-n/2, n/2]) was needed.
        output_wave: Tensor = self.mulaw_dec(output_mu_law)

        return output_wave
