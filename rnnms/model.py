from typing import Tuple

from torch.tensor import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

from .networks.vocoder import RNN_MS_Vocoder

class RNN_MS(pl.LightningModule):
    """RNN_MS, universal neural vocoder.
    """

    def __init__(
        self,
        dim_mel_freq: int,
        dim_latent: int,
        dim_embedding: int = 256,
        dim_rnn_hidden: int = 896,
        dim_out_fc1: int = 1024,
        bits: int = 10,
        sampling_rate: int = 16000,
        hop_length,
        nc: bool = False,
        device
    ):
        """Set up and save the hyperparams.
        """
        super().__init__()

        # params
        self.hparams = {
            "learning_rate": 4.0 * 1e-4,
            "sampling_rate": sampling_rate,
            "sched_decay_rate": 0.5,
            "sched_decay_step": 25000
        }
        self.save_hyperparameters()

        self.rnnms = RNN_MS_Vocoder(
            dim_mel_freq,
            dim_latent,
            dim_embedding,
            dim_rnn_hidden,
            dim_out_fc1,
            bits,
            hop_length,
            nc,
            device
        )

    def forward(self, x: Tensor, mels: Tensor):
        pass

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int):
        """Supervised learning.
        """

        wave_mu_law, spec_mel = batch

        bits_evergy_sereis = self.rnnms(wave_mu_law[:, :-1], spec_mel)
        loss = F.cross_entropy(bits_evergy_sereis.transpose(1, 2), wave_mu_law[:, 1:])

        return {"loss": loss}

    def configure_optimizers(self):
        """Set up a optimizer
        """

        optim = Adam(self.rnnms.parameters(), lr=self.hparams["learning_rate"])
        sched = {
            "scheduler": StepLR(optim, self.hparams["sched_decay_step"], self.hparams["sched_decay_rate"]),
            "interval": "step",
        }

        return [optim], [sched]