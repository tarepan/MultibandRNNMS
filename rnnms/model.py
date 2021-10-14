from typing import Tuple
from dataclasses import dataclass

from torch import no_grad, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from omegaconf import MISSING

from .networks.vocoder import RNN_MS_Vocoder, ConfRNN_MS_Vocoder

"""
    sampling_rate: 16000
    vocoder:
        size_mel_freq: 80
        size_latent: 128
        bits_mu_law: 10
        hop_length: 200
        wave_ar:
            # size_i_cnd: local sync
            size_i_embed_ar: 256
            size_h_rnn: 896
            size_h_fc: 1024
            # size_o_bit: local sync
    optim:
        learning_rate: 4.0 * 1e-4
        sched_decay_rate: 0.5
        sched_decay_step: 25000
"""

@dataclass
class ConfOptim:
    """Configuration of optimizer.
    """
    learning_rate: float = MISSING     # Optimizer learning rate
    sched_decay_rate: float = MISSING  # LR shaduler decay rate
    sched_decay_step: int = MISSING    # LR shaduler decay step

@dataclass
class ConfRNN_MS:
    """Configuration of RNN_MS.
    """
    sampling_rate: int = MISSING  # Audio sampling rate
    vocoder: ConfRNN_MS_Vocoder = ConfRNN_MS_Vocoder()
    optim: ConfOptim = ConfOptim()

class RNN_MS(pl.LightningModule):
    """RNN_MS, universal neural vocoder.
    """

    def __init__(self, conf: ConfRNN_MS):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.rnnms = RNN_MS_Vocoder(conf.vocoder)

    def forward(self, _: Tensor, mels: Tensor):
        return self.rnnms.generate(mels)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        """Supervised learning.
        """

        wave_mu_law, spec_mel = batch

        bits_energy_sereis = self.rnnms(wave_mu_law[:, :-1], spec_mel)
        loss = F.cross_entropy(bits_energy_sereis.transpose(1, 2), wave_mu_law[:, 1:])

        self.log('loss', loss)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        """full length needed & padding is not good (long white seems to be not good for RNN) => cannot batch (batch=1)
        """

        _, mels = batch

        # loss calculation
        # For validation, AR generation can be applied, so cannot use `training_step`.
        # sampling+ARでbits_energy_seriesを作りつつ、サンプルじゃなくてそいつらを評価?
        # o_G = self.training_step(batch, batch_idx, 0)

        # sample generation
        wave = self.rnnms.generate(mels)

        # [-1, 1] restriction
        #   approach A: Clip (x>1 => x=1)
        #   approach B: Scale (max>1 => series/max)
        # In this implementation, already scaled in [-1, 1].

        # [PyTorch](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)
        # add_audio(tag: str, snd_tensor: Tensor(1, L), global_step: Optional[int] = None, sample_rate: int = 44100)
        self.logger.experiment.add_audio(
            f"audio_{batch_idx}",
            wave,
            global_step=self.global_step,
            sample_rate=self.conf.sampling_rate,
        )

        return {
            "val_loss": 0,
        }

    def configure_optimizers(self):
        """Set up a optimizer
        """
        conf = self.conf.optim

        optim = Adam(self.rnnms.parameters(), lr=conf.learning_rate)
        sched = {
            "scheduler": StepLR(optim, conf.sched_decay_step, conf.sched_decay_rate),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }
