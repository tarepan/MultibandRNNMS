from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from omegaconf import MISSING


@dataclass
class ConfRecurrentPreNet:
    """Configuration of RecurrentPreNet.
    Args:
        dim_i: Dimension of input conditioning series
        dim_o: Dimension of output latent series
        num_layers: Number of RNN layers
        bidirectional: Whether RNN is bidirectional or not
    """
    dim_i: int = MISSING
    dim_o: int = MISSING
    num_layers: int = MISSING
    bidirectional: bool = MISSING

class RecurrentPreNet(nn.Module):
    """Transform conditioning input to latent representation through RNN.

    conditioning_input -> (GRU) -> latent_representation
    """
    def __init__(self, conf: ConfRecurrentPreNet):
        super().__init__()

        # Hidden size adjustment: If bidirectional, output dimension become twice
        dim_h = conf.dim_o // 2 if conf.bidirection else conf.dim_o

        self.net = nn.GRU(
            conf.dim_i,
            dim_h,
            num_layers=conf.num_layers,
            batch_first=True,
            bidirectional=conf.bidirectional,
        )

    def forward(self, series: Tensor) -> Tensor:
        """Forward computation for training.

        Args:
            series: Conditioning input series
        Returns:
            Latent representation series
        """
        # (B, T, dim_i) => (B, T, dim_h)
        latent_series, _ = self.net(series)
        return latent_series
