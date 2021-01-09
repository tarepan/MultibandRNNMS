import torch
from torch.tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MuLawDecoding

from .decoder import C_eAR_GenRNN

class RNN_MS_Vocoder(nn.Module):
    """RNN_MS: Universal Vocoder
    """
    def __init__(
        self,
        dim_mel_freq: int,
        size_latent: int,
        size_embedding: int,
        size_rnn_h: int,
        size_fc_h: int,
        bits: int,
        hop_length,
        nc:bool,
        device
    ):
        """RNN_MS

        Args:
            mel_channels: 
            conditioning_channels: 
            nc: If True, mel-spec conditioning is OFF
        """

        super().__init__()
        self.rnn_channels = size_rnn_h
        self.quantization_channels = 2**bits
        self.hop_length = hop_length

        # switch rrn1 based on uc flag
        if nc == True:
            self.condNet = FakeGRU0(dim_mel_freq, size_latent, device, True)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
            print("--------- Mode: no mel-spec conditioning ---------")
        else:
            self.condNet = SpecEncoder(dim_mel_freq, size_latent, hop_length)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
        self.decoder = C_eAR_GenRNN(size_latent, size_embedding, size_rnn_h, size_fc_h, 2**bits)
        self.mulaw_dec = MuLawDecoding(2**bits)

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
