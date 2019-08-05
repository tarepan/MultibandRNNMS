import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchaudio.transforms import MuLawDecoding

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


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class SpecEncoder(nn.Module):
    """2-layer bidirectional GRU mel-spectrogram encoder
    """
    def __init__(self, dim_mel_freq, dim_latent, hop_length):
        super().__init__()
        self.hop_length = hop_length
        self.rnn1 = nn.GRU(dim_mel_freq, dim_latent, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x, mels):
        """forward computation for training. Mel-Spec => latent series.
        Arguments:
            mels {Tensor(Batch, Time, Freq)} -- preprocessed mel-spectrogram
        Returns:
            Tensor(Batch, Time, 2 * dim_latent) -- time-series output latents
        """
        dim_mels_time = mels.size(1)
        audio_slice_frames = x.size(1) // self.hop_length
        pad = (dim_mels_time - audio_slice_frames) // 2

        mels, _ = self.rnn1(mels)
        mels = mels[:, pad:pad + audio_slice_frames, :]
        return mels

    def generate(self, mels):
        mels, _ = self.rnn1(mels)
        return mels


class SimpleAR(nn.Module):
    """Simple 1-layer GRU AutoRegressive Generative Model
    """
    def __init__(self, conditioning_channels, embedding_dim, rnn_channels, fc_channels, bits):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.embedding = nn.Embedding(self.quantization_channels, embedding_dim)
        self.rnn2 = nn.GRU(embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)
        self.mulaw_dec = MuLawDecoding(self.quantization_channels)

    def forward(self, x, latents):
        """forward computation for training
        
        Arguments:
            x -- u-law encoded utterance for self-supervised learning
            latents {Tensor(Batch, Time, Freq)} -- preprocessed mel-spectrogram latent time series
        
        Returns:
            Tensor(Batch, Time, before_softmax) -- time-series output
        """
        x = self.embedding(x)
        x, _ = self.rnn2(torch.cat((x, latents), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, latents):
        output = []
        cell = get_gru_cell(self.rnn2)
        # initialization
        batch_size, sample_size, _ = latents.size()
        h = torch.zeros(batch_size, self.rnn_channels, device=latents.device)
        x = torch.zeros(batch_size, device=latents.device).fill_(self.quantization_channels // 2).long()

        # Manual AR Loop
        # separate speech-conditioning according to Time
        conditionings = torch.unbind(latents, dim=1)
        for m in conditionings:
            x = self.embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            posterior = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(posterior)
            x = dist.sample()
            output.append(2 * x.float().item() / (self.quantization_channels - 1.) - 1.)

        output = self.mulaw_dec(output)
        return output


class RNN_MS(nn.Module):
    """RNN_MS: Universal Vocoder
    """
    def __init__(self, mel_channels, conditioning_channels, embedding_dim,
                 rnn_channels, fc_channels, bits, hop_length, nc:bool, device):
        """RNN_MS
        Arguments:
            nc {bool} -- if True, mel-spec conditioning is OFF
        """
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length

        # switch rrn1 based on uc flag
        if nc == True:
            self.condNet = FakeGRU0(mel_channels, conditioning_channels, device, True)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
            print("--------- Mode: no mel-spec conditioning ---------")
        else:
            self.condNet = SpecEncoder(mel_channels, conditioning_channels, hop_length)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
        self.AR_net = SimpleAR(conditioning_channels, embedding_dim, rnn_channels, fc_channels, bits)

    def forward(self, x, mels):
        """forward computation for training
        
        Arguments:
            x -- u-law encoded utterance for self-supervised learning
            mels {Tensor(Batch, Time, Freq)} -- preprocessed mel-spectrogram
        
        Returns:
            Tensor(Batch, Time, before_softmax) -- time-series output
        """
        # Encoding for conditioning
        latents = self.condNet(x, mels)

        # Cond. Upsampling
        latents = F.interpolate(latents.transpose(1, 2), scale_factor=self.hop_length)
        latents = latents.transpose(1, 2)

        # Autoregressive
        x = self.AR_net(x, latents)

        return x

    def generate(self, mel):
        self.eval()
        with torch.no_grad():
            # Encoding for conditioning
            latents = self.condNet.generate(mel)

            # Cond. Upsampling
            latents = F.interpolate(latents.transpose(1, 2), scale_factor=self.hop_length)
            latents = latents.transpose(1, 2)

            # Autoregressive
            output = self.AR_net.generate(latents)
        self.train()
        return output
