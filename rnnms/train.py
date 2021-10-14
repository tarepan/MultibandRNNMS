from typing import Optional
from enum import Enum
import os
from datetime import timedelta
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cloud_io import get_filesystem
from omegaconf import MISSING

from .model import RNN_MS, ConfRNN_MS


"""
seed: 1234
data:
    batch_size: 32
    num_workers: int = MISSING
    pin_memory: bool = MISSING
    adress_data_root: str = MISSING
train:
    ckptLog:
        dir_root: logs
        name_exp: default
        name_version: version_-1
    trainer:
        max_epochs: 500
        val_interval_epoch: 4
        profiler:
    model:
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

class Profiler(Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"


@dataclass
class ConfTrainer:
    """Configuration of trainer.
    """
    max_epochs: int = MISSING
    val_interval_epoch: int = MISSING
    profiler: Profiler = MISSING

@dataclass
class ConfCkptLog:
    """Configuration of checkpointing and logging.
    """
    dir_root = MISSING
    name_exp = MISSING
    name_version = MISSING

@dataclass
class ConfTrain:
    """Configuration of train.
    """
    ckpt_log: ConfCkptLog = ConfCkptLog()
    trainer: ConfTrainer = ConfTrainer()
    model: ConfRNN_MS = ConfRNN_MS()

def train(conf: ConfTrain, datamodule: LightningDataModule) -> None:
    """Train RNN_MS on PyTorch-Lightning.
    """

    # [todo]: Use snake_case
    ckpt_and_logging = CheckpointAndLogging(conf.ckpt_log.dir_root, conf.ckpt_log.name_exp, conf.ckpt_log.name_version)
    # setup
    model = RNN_MS(conf.model)

    # Save checkpoint as `last.ckpt` every 15 minutes.
    ckpt_cb = ModelCheckpoint(
        train_time_interval=timedelta(minutes=15),
        save_last=True,
        save_top_k=0,
    )

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        auto_select_gpus=True,
        precision=16,
        max_epochs=conf.trainer.max_epochs,
        check_val_every_n_epoch=conf.trainer.val_interval_epoch,
        # logging/checkpointing
        resume_from_checkpoint=ckpt_and_logging.resume_from_checkpoint,
        default_root_dir=ckpt_and_logging.default_root_dir,
        logger=pl_loggers.TensorBoardLogger(
            ckpt_and_logging.save_dir, ckpt_and_logging.name, ckpt_and_logging.version
        ),
        callbacks=[ckpt_cb],
        # reload_dataloaders_every_epoch=True,
        profiler=conf.trainer.profiler,
        progress_bar_refresh_rate=30
    )

    # training
    trainer.fit(model, datamodule=datamodule)


class CheckpointAndLogging:
    """Generate path of checkpoint & logging.
    {dir_root}/
        {name_exp}/
            {name_version}/
                checkpoints/
                    {name_ckpt} # PyTorch-Lightning Checkpoint. Resume from here.
                hparams.yaml
                events.out.tfevents.{xxxxyyyyzzzz} # TensorBoard log file.
    """

    # [PL's Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api)
    default_root_dir: Optional[str]
    resume_from_checkpoint: Optional[str]
    # [PL's TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/stable/logging.html#tensorboard)
    save_dir: str
    name: str
    version: str
    # [PL's ModelCheckpoint](https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)
    # dirpath: Implicitly inferred from `default_root_dir`, `name` and `version` by PyTorch-Lightning

    def __init__(
        self,
        dir_root: str,
        name_exp: str = "default",
        name_version: str = "version_-1",
        name_ckpt: str = "last.ckpt",
    ) -> None:

        path_ckpt = os.path.join(dir_root, name_exp, name_version, "checkpoints", name_ckpt)

        # PL's Trainer
        self.default_root_dir = dir_root
        self.resume_from_checkpoint = path_ckpt if get_filesystem(path_ckpt).exists(path_ckpt) else None

        # TB's TensorBoardLogger
        self.save_dir = dir_root
        self.name = name_exp
        self.version = name_version