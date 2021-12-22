"""Train RNNMS"""


from typing import Optional
from enum import Enum
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from omegaconf import MISSING

from .model import MbRNNMS, ConfMbRNNMS
from .report import ConfCkptLog, generate_state_components


class Profiler(Enum):
    """PyTorch-Lightning's Profiler types"""
    SIMPLE = "simple"
    ADVANCED = "advanced"


@dataclass
class ConfTrainer:
    """Configuration of trainer.
    Args:
        max_epochs: Number of maximum training epoch
        val_interval_epoch: Interval epoch between validation
        profiler: Profiler setting
    """
    max_epochs: int = MISSING
    val_interval_epoch: int = MISSING
    profiler: Optional[Profiler] = MISSING


@dataclass
class ConfTrain:
    """Configuration of train.
    """
    ckpt_log: ConfCkptLog = ConfCkptLog()
    trainer: ConfTrainer = ConfTrainer()
    model: ConfMbRNNMS = ConfMbRNNMS()


def train(conf: ConfTrain, datamodule: LightningDataModule) -> None:
    """Train RNN_MS on PyTorch-Lightning.
    """

    callbacks = []

    # setup
    model = MbRNNMS(conf.model)

    # Resume and Reporting
    state_components, state_clbks = generate_state_components(conf.ckpt_log)
    callbacks.extend(state_clbks)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        auto_select_gpus=True,
        precision=16,
        max_epochs=conf.trainer.max_epochs,
        check_val_every_n_epoch=conf.trainer.val_interval_epoch,
        callbacks=callbacks,
        # reload_dataloaders_every_epoch=True,
        profiler=conf.trainer.profiler,
        progress_bar_refresh_rate=30,
        **state_components
    )

    # training
    trainer.fit(model, datamodule=datamodule)
