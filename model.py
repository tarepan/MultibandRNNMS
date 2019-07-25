import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import mulaw_decode
from tqdm import tqdm


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


class Vocoder(nn.Module):
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
            self.rnn1 = FakeGRU0(mel_channels, conditioning_channels, True, device)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
            print("--------- Mode: no mel-spec conditioning ---------")
        else:
            self.rnn1 = nn.GRU(mel_channels, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True)
            # output: (batch, seq_len, 2 * conditioning_channels), h_n
        self.embedding = nn.Embedding(self.quantization_channels, embedding_dim)
        self.rnn2 = nn.GRU(embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, mels):
        """
            mels {Tensor(Batch, Time, Freq)}
        """
        sample_frames = mels.size(1)
        audio_slice_frames = x.size(1) // self.hop_length
        pad = (sample_frames - audio_slice_frames) // 2

        # Conditioning
        mels, _ = self.rnn1(mels)
        mels = mels[:, pad:pad + audio_slice_frames, :]

        mels = F.interpolate(mels.transpose(1, 2), scale_factor=self.hop_length)
        mels = mels.transpose(1, 2)

        x = self.embedding(x)

        # AR
        x, _ = self.rnn2(torch.cat((x, mels), dim=2))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, mel):
        self.eval()

        output = []
        cell = get_gru_cell(self.rnn2)

        with torch.no_grad():
            # Conditioning
            mel, _ = self.rnn1(mel)

            mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
            mel = mel.transpose(1, 2)

            batch_size, sample_size, _ = mel.size()

            h = torch.zeros(batch_size, self.rnn_channels, device=mel.device)
            x = torch.zeros(batch_size, device=mel.device).fill_(self.quantization_channels // 2).long()

            conditionings = torch.unbind(mel, dim=1)
            # AR
            for m in conditionings:
                x = self.embedding(x)
                h = cell(torch.cat((x, m), dim=1), h)

                x = F.relu(self.fc1(h))
                logits = self.fc2(x)

                posterior = F.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(posterior)

                x = dist.sample()
                output.append(2 * x.float().item() / (self.quantization_channels - 1.) - 1.)

        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)

        self.train()
        return output
