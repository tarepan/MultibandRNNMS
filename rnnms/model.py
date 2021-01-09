from typing import Tuple

from torch.tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl


class RNN_MS(pl.LightningModule):
    """RNN_MS, universal neural vocoder.
    """

    def __init__(
        self,
        dim_mel_freq: int,
        dim_latent: int,
        dim_embedding: int,
        dim_rnn_hidden: int,
        dim_out_fc1: int,
        bits: int,
        hop_length,
        nc:bool,
        device
    ):
        super().__init__()

        # params
        self.hparams = {
            "learning_rate": 2.0 * 1e-4,
            "sampling_rate": sampling_rate,
            "sched_decay_rate": "",
            "sched_decay_step": ""
        }
        self.save_hyperparameters()

        self.rnnms = RNN_MS(dim_mel_freq, dim_latent, dim_embedding, dim_rnn_hidden, dim_out_fc1, bits, hop_length, hc, device)

    def forward(self, x: Tensor, mels: Tensor) -> Tensor:
        pass

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int):
        """Supervised learning.
        """

        wave_ml, spec_mel = batch

        wave_ml_gen = self.rnnms(wave_ml[:, :-1], spec_mel)
        loss = F.cross_entropy(wave_ml_gen.transpose(1, 2), wave_ml[:, 1:])

        return {"loss": loss}

    def configure_optimizers(self):
        """Set up a optimizer
        """

        optim = Adam(self.rnnms.parameters(), lr=params["vocoder"]["learning_rate"])
        sched = {
            "scheduler": StepLR(optim, decay_iter, decay_rate),
            "interval": "step",
        }

        return [optim], [sched]