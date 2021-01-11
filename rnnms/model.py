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
        size_mel_freq: int = 80,
        hop_length: int = 200,
        sampling_rate: int = 16000,
        size_latent: int = 128,
        size_embed_ar: int = 256,
        size_rnn_h: int = 896,
        size_fc_h: int = 1024,
        bits_mu_law: int = 10,
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
            size_mel_freq,
            size_latent,
            size_embed_ar,
            size_rnn_h,
            size_fc_h,
            bits_mu_law,
            hop_length,
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