"""Vocoder networks"""


from dataclasses import dataclass

from torch import Tensor, cat # pylint: disable=no-name-in-module
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MuLawDecoding
from omegaconf import MISSING

from .prenet import RecurrentPreNet, ConfRecurrentPreNet
from .decoder import CeARSubGenRNN, ConfCeARGenRNN
from .pqmf import PQMF

@dataclass
class ConfRNNMSVocoder:
    """Configuration of RNNMSVocoder.
    Args:
        dim_i_feature: Dimension of input feature (e.g. Frequency dim of mel-spec)
        dim_voc_latent: Dimension of vocoder's latent between PreNet and WaveRNN
        bits_mu_law: Bit depth of μ-law encoding
        upsampling_t: Factor of time-direcitonal latent upsampling (e.g. STFT stride)
        n_band: Number of subbands in sample AR
    """
    dim_i_feature: int = MISSING
    dim_voc_latent: int = MISSING
    bits_mu_law: int = MISSING
    upsampling_t: int = MISSING
    n_band: int = MISSING
    prenet: ConfRecurrentPreNet = ConfRecurrentPreNet(
        dim_i="${..dim_i_feature}",
        dim_o="${..dim_voc_latent}")
    wave_ar: ConfCeARGenRNN = ConfCeARGenRNN(
        size_i_cnd="${..dim_voc_latent}",
        size_o_bit="${..bits_mu_law}")

class RNNMSVocoder(nn.Module):
    """RNN_MS Universal Vocoder, generating speech from conditioning.
    """
    def __init__(self, conf: ConfRNNMSVocoder):
        super().__init__()
        self._time_upsampling_factor = conf.upsampling_t
        self._n_band = conf.n_band

        # PreNet which transform conditioning inputs into latent representation
        self._prenet = RecurrentPreNet(conf.prenet)
        # AR which yeild sample probability autoregressively
        self._ar = CeARSubGenRNN(conf.wave_ar)
        self._mulaw_dec = MuLawDecoding(2**conf.bits_mu_law)
        self._pqmf = PQMF()

    def forward(self, wave_mu_law: Tensor, cond_series: Tensor) -> Tensor:
    # def forward(self, cond_series: Tensor, wave_mu_law: Tensor) -> Tensor:
        """Generate bit-energy series with teacher-forcing.

        Args:
            wave_mu_law (Batch, Band, T_wave): mu-law waveforms (index series) for teacher-forcing
            cond_series (Batch, T_cond, Feat): conditioning series
        Returns:
            (Batch, Band, T_wave, Energy) Generated mu-law encoded waveforms
        """
        # Conditioning to Latent :: (B, T_cond, Feat) => (B, T_cond, Latent)
        latents = self._prenet(cond_series)

        # Latent Time Upsampling :: (B, T_cond, Latent) => (B, T_sub, Latent)
        latents_upsampled: Tensor = F.interpolate(
            latents.transpose(1, 2),
            scale_factor=self._time_upsampling_factor // self._n_band
        ).transpose(1, 2)

        # Sample AR :: (B, T_sub, Latent) => (B, Band, T_sub, Energy)
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
            scale_factor=self._time_upsampling_factor // self._n_band
        ).transpose(1, 2)

        # Sample subband μ-law waveforms :: (B=1, Band, T_sub), sample ∈ {x∈N | 0<=x<=2^bit -1}
        subbands_mu_law = self._ar.generate(latents_upsampled)

        # μ-law expansion.
        # range: [0, 2^bit -1] => [-1, 1]
        # [PyTorch](https://pytorch.org/audio/stable/transforms.html#mulawdecoding)
        # :: (B=1, T_sub) xBand => (B=1, 1, T_sub) xBand => (B=1, Band, T_sub)
        raw_b1: Tensor = self._mulaw_dec(subbands_mu_law[:, 0]).unsqueeze(1)
        raw_b2: Tensor = self._mulaw_dec(subbands_mu_law[:, 1]).unsqueeze(1)
        raw_b3: Tensor = self._mulaw_dec(subbands_mu_law[:, 2]).unsqueeze(1)
        raw_b4: Tensor = self._mulaw_dec(subbands_mu_law[:, 3]).unsqueeze(1)
        raw_bands = cat((raw_b1, raw_b2, raw_b3, raw_b4), dim=1)

        # Combine subbands :: (B=1, Band, T_sub) => (B=1, Band=1, T_wave) => (B=1, T_wave)
        fullband = self._pqmf.synthesis(raw_bands).view(1, -1)
        return fullband
