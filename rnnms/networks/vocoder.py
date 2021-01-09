import torch
from torch.tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MuLawDecoding

from .encoder import FakeGRU0, SpecEncoder
from .decoder import C_eAR_GenRNN


class RNN_MS_Vocoder(nn.Module):
    """RNN_MS: Universal Vocoder
    """
    def __init__(
        self,
        size_mel_freq: int,
        size_latent: int,
        size_embed_ar: int,
        size_rnn_h: int,
        size_fc_h: int,
        bits_mu_law: int,
        hop_length,
        nc:bool,
        device
    ):
        """Set up the hyperparams.

        Args:
            size_mel_freq: size of mel frequency dimension
            size_latent: size of latent vector
            size_embed_ar: size of embedded auto-regressive input vector (embedded output_t-1)
            size_rnn_h: size of decoder's RNN hidden vector
            size_fc_h: size of decoder's FC hidden layer
            bits_mu_law: bit depth of μ-law encoding
            hop_length:
            nc: If True, mel-spec conditioning is OFF
            device:
        """

        super().__init__()
        self.rnn_channels = size_rnn_h
        self.hop_length = hop_length

        # switch rrn1 based on uc flag
        if nc == True:
            self.condNet = FakeGRU0(size_mel_freq, size_latent, device, True)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
            print("--------- Mode: no mel-spec conditioning ---------")
        else:
            self.condNet = SpecEncoder(size_mel_freq, size_latent, hop_length)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
        self.decoder = C_eAR_GenRNN(size_latent, size_embed_ar, size_rnn_h, size_fc_h, 2**bits_mu_law)
        self.mulaw_dec = MuLawDecoding(2**bits_mu_law)

    def forward(self, wave_mu_law: Tensor, mels: Tensor) -> Tensor:
        """Forward computation for training.
        
        Arguments:
            wave_mu_law: mu-law encoded waveform
            mels (Tensor(Batch, Time, Freq)): preprocessed mel-spectrogram
        
        Returns:
            Tensor(Batch, Time, before_softmax) Generated mu-law encoded waveform
        """
        # Encoding for conditioning
        latents = self.condNet(wave_mu_law, mels)

        # Cond. Upsampling
        latents = F.interpolate(latents.transpose(1, 2), scale_factor=self.hop_length)
        latents = latents.transpose(1, 2)

        # Autoregressive
        wave_mu_law = self.decoder(wave_mu_law, latents)

        return wave_mu_law

    def generate(self, mel):
        self.eval()
        with torch.no_grad():
            # Encoding for conditioning
            latents = self.condNet.generate(mel)

            # Cond. Upsampling
            latents = F.interpolate(latents.transpose(1, 2), scale_factor=self.hop_length)
            latents = latents.transpose(1, 2)

            # Autoregressive
            output_mu_law = self.decoder.generate(latents)
            output = self.mulaw_dec(output_mu_law)
        self.train()
        return output
