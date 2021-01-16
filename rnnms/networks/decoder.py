import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def get_gru_cell(gru):
    """Transfer (learned) GRU state to a new GRUCell.
    """

    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class C_eAR_GenRNN(nn.Module):
    """Latent-conditional, embedded-auto-regressive Generative RNN.

    conditioning latent + embeded sample_t-1 -> (RNN) -> (FC) -->
      --> output Energy vector -> (softmax) -> (sampling) -> sample_t

    Alternative implementation: embedding => one-hot
    """

    def __init__(self, size_i_cnd: int, size_i_embed_ar: int, size_h_rnn: int, size_h_fc: int, size_o: int) -> None:
        """Set up the hyperparams.

        Args:
            size_i_cnd: size of conditioning input vector
            size_i_embed_ar: size of embedded auto-regressive input vector (embedded sample_t-1)
            size_h_rnn: size of RNN hidden vector
            size_h_fc: size of 2-layer FC's hidden layer
            size_o: size of output energy vector
        """

        super().__init__()

        # output embedding
        self.size_out = size_o
        self.embedding = nn.Embedding(size_o, size_i_embed_ar)

        # RNN module
        self.size_h_rnn = size_h_rnn
        self.rnn = nn.GRU(size_i_embed_ar + 2 * size_i_cnd, size_h_rnn, batch_first=True)

        # RNN_out -> μ-law bits energy
        self.fc1 = nn.Linear(size_h_rnn, size_h_fc)
        self.fc2 = nn.Linear(size_h_fc, size_o)

    def forward(self, reference_sample: Tensor, i_cnd_series: Tensor) -> Tensor:
        """Forward for training.
        
        Forward RNN computation for training.
        This is for training, so there is no sampling.
        For training, auto-regressive input is replaced by self-supervised input (`reference_sample`).

        Args:
            reference_sample: Reference sample (series) for for self-supervised learning
            i_cnd_series (Tensor(Batch, Time, dim_latent)): conditional input vector series
        
        Returns:
            (Tensor(Batch, Time, 2*bits)) Series of output energy vector
        """

        # Embed whole reference series (non-AR) because this is training.
        sample_ref_emb = self.embedding(reference_sample)
        o_rnn, _ = self.rnn(torch.cat((sample_ref_emb, i_cnd_series), dim=2))
        o = self.fc2(F.relu(self.fc1(o_rnn)))
        return o

    def generate(self, i_cnd_series: Tensor) -> Tensor:
        """
        Generate samples auto-regressively with given latent series.

        Returns:
            Sample series, each point is in range [0, (int), size_o - 1]
        """

        # Temporal care for OOM by long audio inference
        # l = i_cnd_series.size(1)
        # if l > 3000:
        #     i_cnd_series = i_cnd_series[:, :3000, :]

        batch_size = i_cnd_series.size(0)
        # [Batch, T] (initialized as [Batch, 0])
        sample_series = torch.tensor([[] for _ in range(batch_size)], device=i_cnd_series.device)
        cell = get_gru_cell(self.rnn)
        # initialization
        h_prev = torch.zeros(batch_size, self.size_h_rnn, device=i_cnd_series.device)
        # [Batch]
        # nn.Embedding needs LongTensor input
        sample_t_minus_1 = torch.zeros(batch_size, device=i_cnd_series.device, dtype=torch.long)
        # ※ μ-law specific part
        # In μ-law representation, center == volume 0, so self.size_out // 2 equal to zero volume
        sample_t_minus_1 = sample_t_minus_1.fill_(self.size_out // 2)

        # Auto-regiressive sample series generation
        # separate speech-conditioning according to Time
        # [Batch, T_mel, freq] => [Batch, freq]
        conditionings = torch.unbind(i_cnd_series, dim=1)
        i = 0
        for i_cond_t in conditionings:
            # [Batch] => [Batch, size_i_embed_ar]
            i_embed_ar_t = self.embedding(sample_t_minus_1)
            h_rnn_t = cell(torch.cat((i_embed_ar_t, i_cond_t), dim=1), h_prev)
            o_t = self.fc2(F.relu(self.fc1(h_rnn_t)))
            posterior_t = F.softmax(o_t, dim=1)
            dist_t = torch.distributions.Categorical(posterior_t)
            # Random sampling from categorical distribution
            sample_t: Tensor = dist_t.sample()
            # Reshape: [Batch] => [Batch, 1] (can be concatenated with [Batch, T])
            sample_series = torch.cat((sample_series, sample_t.reshape((-1, 1))), dim=1)
            sample_t_minus_1 = sample_t
            print(i)
            print(sample_series.size())
            i = i+1

        return sample_series
