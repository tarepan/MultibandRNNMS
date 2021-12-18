"""Decoder networks"""


from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING


def get_gru_cell(gru):
    """Transfer (learned) GRU state to a new GRUCell.
    """

    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


@dataclass
class ConfC_eAR_GenRNN:
    """Configuration of C_eAR_GenRNN.
    Args:
        size_i_cnd: size of conditioning input vector
        size_i_embed_ar: size of embedded auto-regressive input vector (embedded sample_t-1)
        size_h_rnn: size of RNN hidden vector
        size_h_fc: size of 2-layer FC's hidden layer
        size_o: size of output energy vector
    """
    size_i_cnd: int = MISSING
    size_i_embed_ar: int = MISSING
    size_h_rnn: int = MISSING
    size_h_fc: int = MISSING
    size_o_bit: int = MISSING

class CeARSubGenRNN(nn.Module):
    """Latent-conditional, embedded-auto-regressive Generative RNN with subbands.

    conditioning latent + embeded sample_bN_t-1 -> (RNN) -> (FC1) -->
      --> FC2 xBand -> output Energy vector -> (softmax) -> (sampling) -> sample_bN_t
    """

    def __init__(self, conf: ConfC_eAR_GenRNN) -> None:
        super().__init__()

        self._n_band = 4

        # output embedding
        self.size_out = 2**conf.size_o_bit
        self.embedding_b1 = nn.Embedding(self.size_out, conf.size_i_embed_ar)
        self.embedding_b2 = nn.Embedding(self.size_out, conf.size_i_embed_ar)
        self.embedding_b3 = nn.Embedding(self.size_out, conf.size_i_embed_ar)
        self.embedding_b4 = nn.Embedding(self.size_out, conf.size_i_embed_ar)
        dim_emb = self._n_band * conf.size_i_embed_ar

        # RNN module: Embedded_sample_t-1 + latent_t-1 => hidden_t
        self.size_h_rnn = conf.size_h_rnn
        self.rnn = nn.GRU(dim_emb + conf.size_i_cnd, conf.size_h_rnn, batch_first=True)

        # FC module: RNN_out => M-band μ-law bits energy
        ## Common FC-ReLU
        self.fc1 = nn.Sequential(nn.Linear(conf.size_h_rnn, conf.size_h_fc), nn.ReLU())
        ## Band-specific FC
        self.fc2_b1 = nn.Linear(conf.size_h_fc, self.size_out)
        self.fc2_b2 = nn.Linear(conf.size_h_fc, self.size_out)
        self.fc2_b3 = nn.Linear(conf.size_h_fc, self.size_out)
        self.fc2_b4 = nn.Linear(conf.size_h_fc, self.size_out)

    def forward(self, reference_sample: Tensor, i_cnd_series: Tensor) -> Tensor:
        """Forward for training.

        Forward RNN computation for training with teacher-forcing.
        This is for training, so there is no sampling.

        Args:
            reference_sample (Batch, Band, Time): Reference sample (index) series for teacher-forcing
            i_cnd_series (Batch, Time, dim_latent): conditional input vector series

        Returns:
            (Batch, Band, Time, 2*bits): Series of output energy vector
        """

        # Embed teacher-forcing signals: (B, Band, T) => (B, T) xBand => (B, T, Emb) xBand
        ref_emb_b1 = self.embedding_b1(reference_sample[:, 0])
        ref_emb_b2 = self.embedding_b2(reference_sample[:, 1])
        ref_emb_b3 = self.embedding_b3(reference_sample[:, 2])
        ref_emb_b4 = self.embedding_b4(reference_sample[:, 3])
        # Concat embeddings: (B, T, Emb) xBand => (B, T, Band*Emb)
        ref_emb_series = torch.cat((ref_emb_b1, ref_emb_b2, ref_emb_b3, ref_emb_b4), dim=-1)

        # Concat RNN inputs: (B, T, Band*Emb) + (B, T, Feat) => (B, T, Band*Emb+Feat)
        i_ar = torch.cat((ref_emb_series, i_cnd_series), dim=-1)
        # (B, T, Band*Emb+Feat) => (B, T, h_rnn)
        o_rnn, _ = self.rnn(i_ar)

        # (B, T, h_rnn) => (B, T, h_fc)
        o_fc1 = self.fc1(o_rnn)
        # (B, T, h_fc) => (B, T, Energy) xBand => (B, 1, T, Energy) xBand
        o_b1 = self.fc2_b1(o_fc1).unsqueeze(1)
        o_b2 = self.fc2_b2(o_fc1).unsqueeze(1)
        o_b3 = self.fc2_b3(o_fc1).unsqueeze(1)
        o_b4 = self.fc2_b4(o_fc1).unsqueeze(1)
        # (B, 1, T, Energy) xBand => (B, Band, T, Energy)
        o_bands = torch.cat((o_b1, o_b2, o_b3, o_b4), 1)
        return o_bands

    def generate(self, i_cnd_series: Tensor) -> Tensor:
        """
        Generate samples auto-regressively with given latent series.

        Args:
            i_cnd_series (B, T, Latent): Conditioning input series
        Returns:
            (B, Band, T): Sample series, sample ∈ {x∈N | 0<=x<=size_o-1}
        """

        batch_size = i_cnd_series.size(0)
        # ::(B, Band, T) (initialized as (B, Band, 0])
        sample_series = torch.tensor(
            [[[] for _ in range(0, self._n_band)] for _ in range(batch_size)],
            device=i_cnd_series.device
        )
        cell = get_gru_cell(self.rnn)
        # initialization
        h_rnn_t_minus_1 = torch.zeros(batch_size, self.size_h_rnn, device=i_cnd_series.device)
        # [Batch]
        # nn.Embedding needs LongTensor input
        sample_b1_t_minus_1 = torch.zeros(batch_size, device=i_cnd_series.device, dtype=torch.long)
        sample_b2_t_minus_1 = torch.zeros(batch_size, device=i_cnd_series.device, dtype=torch.long)
        sample_b3_t_minus_1 = torch.zeros(batch_size, device=i_cnd_series.device, dtype=torch.long)
        sample_b4_t_minus_1 = torch.zeros(batch_size, device=i_cnd_series.device, dtype=torch.long)
        # ※ μ-law specific part
        # In μ-law representation, center == volume 0, so self.size_out // 2 equal to zero volume
        sample_b1_t_minus_1 = sample_b1_t_minus_1.fill_(self.size_out // 2)
        sample_b2_t_minus_1 = sample_b2_t_minus_1.fill_(self.size_out // 2)
        sample_b3_t_minus_1 = sample_b3_t_minus_1.fill_(self.size_out // 2)
        sample_b4_t_minus_1 = sample_b4_t_minus_1.fill_(self.size_out // 2)

        # Auto-regiressive sample series generation
        # separate speech-conditioning according to Time
        # [Batch, T_mel, freq] => [Batch, freq]
        conditionings = torch.unbind(i_cnd_series, dim=1)

        for i_cond_t in conditionings:
            # (B,) xBand => (B, Emb) xBand => (B, Band*Emb)
            ar_1_t = self.embedding_b1(sample_b1_t_minus_1)
            ar_2_t = self.embedding_b2(sample_b2_t_minus_1)
            ar_3_t = self.embedding_b3(sample_b3_t_minus_1)
            ar_4_t = self.embedding_b4(sample_b4_t_minus_1)
            i_emb_t = torch.cat((ar_1_t, ar_2_t, ar_3_t, ar_4_t), dim=-1)
            # (B, Band*Emb) + (B, Latent) => (B, h_rnn)
            h_rnn_t = cell(torch.cat((i_emb_t, i_cond_t), dim=-1), h_rnn_t_minus_1)
            # (B, h_rnn) => (B, h_fc) => (B, Prob) xBand
            h_fc_t = self.fc1(h_rnn_t)
            posterior_b1_t = F.softmax(self.fc2_b1(h_fc_t), dim=-1)
            posterior_b2_t = F.softmax(self.fc2_b2(h_fc_t), dim=-1)
            posterior_b3_t = F.softmax(self.fc2_b3(h_fc_t), dim=-1)
            posterior_b4_t = F.softmax(self.fc2_b4(h_fc_t), dim=-1)
            dist_b1_t = torch.distributions.Categorical(posterior_b1_t)
            dist_b2_t = torch.distributions.Categorical(posterior_b2_t)
            dist_b3_t = torch.distributions.Categorical(posterior_b3_t)
            dist_b4_t = torch.distributions.Categorical(posterior_b4_t)
            # Sampling from categorical dist.
            # :: (B, Prob) => (B,) xBand => (B, 1) xBand => (B, Band)
            sample_b1_t: Tensor = dist_b1_t.sample().unsqueeze(1)
            sample_b2_t: Tensor = dist_b2_t.sample().unsqueeze(1)
            sample_b3_t: Tensor = dist_b3_t.sample().unsqueeze(1)
            sample_b4_t: Tensor = dist_b4_t.sample().unsqueeze(1)
            sample_t = torch.cat((sample_b1_t, sample_b2_t, sample_b3_t, sample_b4_t), dim=1)
            # Reshape: (B, Band) => (Batch, Band, T=1) for cancat w/ (B, Band, T)
            sample_series = torch.cat((sample_series, sample_t.unsqueeze(-1)), dim=-1)
            # t => t-1 :: (B, 1) => (B)
            sample_b1_t_minus_1 = sample_b1_t.squeeze(-1)
            sample_b2_t_minus_1 = sample_b2_t.squeeze(-1)
            sample_b3_t_minus_1 = sample_b3_t.squeeze(-1)
            sample_b4_t_minus_1 = sample_b4_t.squeeze(-1)
            h_rnn_t_minus_1 = h_rnn_t

        return sample_series
