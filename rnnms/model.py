from typing import Tuple

from torch import no_grad, Tensor
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
        # Hardcoded hyperparams
        size_mel_freq: int = 80,
        hop_length: int = 200,
        sampling_rate: int = 16000,
        size_latent: int = 128,
        size_embed_ar: int = 256,
        size_rnn_h: int = 896,
        size_fc_h: int = 1024,
        bits_mu_law: int = 10,
        learning_rate: float = 4.0 * 1e-4,
        sched_decay_rate: float = 0.5,
        sched_decay_step: int = 25000,
    ):
        """Set up and save the hyperparams.
        """

        super().__init__()
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
            sample_rate=self.hparams.sampling_rate,
        )

        return {
            "val_loss": 0,
        }

    def configure_optimizers(self):
        """Set up a optimizer
        """

        optim = Adam(self.rnnms.parameters(), lr=self.hparams.learning_rate)
        sched = {
            "scheduler": StepLR(optim, self.hparams.sched_decay_step, self.hparams.sched_decay_rate),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }
