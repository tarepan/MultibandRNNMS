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
        self._time_upsampling_factor = conf.upsampling_t

        # PreNet which transform conditioning inputs into latent representation
        self._prenet = RecurrentPreNet(conf.prenet)
        # AR which yeild sample probability autoregressively
        self._ar = C_eAR_GenRNN(conf.wave_ar)
        self._mulaw_dec = MuLawDecoding(2**conf.bits_mu_law)

    def forward(self, wave_mu_law: Tensor, cond_series: Tensor) -> Tensor:
    # def forward(self, cond_series: Tensor, wave_mu_law: Tensor) -> Tensor:
        """Generate bit-energy series with teacher-forcing.

        Args:
            wave_mu_law (Batch, T_wave): mu-law waveforms (index series) for teacher-forcing
            cond_series (Batch, T_cond, Feat): conditioning series

        Returns:
            (Batch, T_wave, Energy) Generated mu-law encoded waveforms
        """
        # Conditioning to Latent :: (B, T_cond, Feat) => (B, T_cond, Latent)
        latents = self._prenet(cond_series)

        # Latent Time Upsampling :: (B, T_cond, Latent) => (B, T_wave, Latent)
        latents_upsampled: Tensor = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=self._time_upsampling_factor
        ).transpose(1, 2)

        # Sample AR :: (B, T_wave, Latent) => (B, T_wave, Energy)
        bits_energy_series = self._ar(wave_mu_law, latents_upsampled)

        return bits_energy_series

    def generate(self, cond_series: Tensor) -> Tensor:
        """Generate waveforms from conditioning series.

        Input has variable length, and long zero padding is not good for RNN,
        so batch_size must be 1.

        Args:
            cond_series (1, T_cond, Feat): Conditioning series
        Returns:
            (1, T_wave): Generated raw waveform. A sample point is in range [-1, 1].
        """

        # Transform conditionings into upsampled latents
        latents = self._prenet(cond_series)
        latents_upsampled: Tensor = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=self._time_upsampling_factor
        ).transpose(1, 2)

        # Sample a waveform (sequence of μ-law encoded samples)
        output_mu_law = self._ar.generate(latents_upsampled)

        # μ-law expansion.
        # range: [0, 2^bit -1] => [-1, 1]
        # [PyTorch](https://pytorch.org/audio/stable/transforms.html#mulawdecoding)
        # bshall/UniversalVocoding use librosa,
        # so range adaption ([0, n] -> [-n/2, n/2]) was needed.
        output_wave: Tensor = self._mulaw_dec(output_mu_law)

        return output_wave
