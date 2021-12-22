"""RNNMS PyTorch-Lightnig model"""


from typing import Tuple
from dataclasses import dataclass

from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from omegaconf import MISSING

from .networks.vocoder import RNNMSVocoder, ConfRNNMSVocoder


@dataclass
class ConfOptim:
    """Configuration of optimizer.
    Args:
        learning_rate: Optimizer learning rate
        sched_decay_rate: LR shaduler decay rate
        sched_decay_step: LR shaduler decay step
    """
    learning_rate: float = MISSING
    sched_decay_rate: float = MISSING
    sched_decay_step: int = MISSING

@dataclass
class ConfMbRNNMS:
    """Configuration of RNN_MS.
    """
    sampling_rate: int = MISSING  # Audio sampling rate
    vocoder: ConfRNNMSVocoder = ConfRNNMSVocoder()
    optim: ConfOptim = ConfOptim()

class MbRNNMS(pl.LightningModule):
    """RNN_MS, universal neural vocoder.
    """

    def __init__(self, conf: ConfMbRNNMS):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.rnnms = RNNMSVocoder(conf.vocoder)

    def forward(self, cond_series: Tensor) -> Tensor:
        """Generate a waveform from conditioning series.

        Intended to be used for ONNX export.
        Args:
            cond_series (Batch=1, Time, Feature) - conditioning series
        Returns:
            Tensor(1, Time') - a waveform
        """
        return self.rnnms.generate(cond_series)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        """Supervised learning.
        """

        # ::(B, Band, T_sub), (B, ?, ?)
        wave_mu_law, spec_mel = batch

        # :: => (Batch, Band, T_sub, Energy)
        bits_energy_sereis = self.rnnms(wave_mu_law[:, :, :-1], spec_mel)
        # CE loss over bands and times
        #   energy :: (B, Band, T_sub, E) => (B, E, T_sub, Band)
        #   GT     :: (B, Band, T_sub)    => (B,    T_sub, Band)
        loss = F.cross_entropy(
            bits_energy_sereis.transpose(1, 3),
            wave_mu_law[:, :, 1:].transpose(1, 2)
        )

        self.log('loss', loss)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        """full length needed & padding is not good (long white seems to be not good for RNN)
        => cannot batch (batch=1)
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
