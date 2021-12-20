"""Report model states through PyTorch-Lightning"""

from typing import Optional
import os
from dataclasses import dataclass
from datetime import timedelta

from omegaconf import MISSING

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import get_filesystem


@dataclass
class ConfCkptLog:
    """Configuration of `CheckpointAndLogging`.
    """
    dir_root: str = MISSING
    name_exp: str  = "default"
    name_version: str  = "version_-1"
    name_ckpt: str = "last.ckpt"

class CheckpointAndLogging:
    """Generate path of checkpoint & logging.

    Generated paths have same names with corresponding PyTorch-Lighting's arguments.
    """
    # The paths:
    # ```
    # {dir_root}/
    #     {name_exp}/
    #         {name_version}/
    #             checkpoints/
    #                 {name_ckpt} # PyTorch-Lightning Checkpoint. Resume from here.
    #             hparams.yaml
    #             events.out.tfevents.{xxxxyyyyzzzz} # TensorBoard log file.
    # ```

    # [PL's Trainer]
    # (https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api)
    default_root_dir: Optional[str]
    resume_from_checkpoint: Optional[str]
    # [PL's TensorBoardLogger]
    # (https://pytorch-lightning.readthedocs.io/en/stable/logging.html#tensorboard)
    save_dir: str
    name: str
    version: str
    # [PL's ModelCheckpoint]
    # (https://pytorch-lightning.readthedocs.io/en/stable/generated/
    # pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint)
    # dirpath: Inferred from `default_root_dir`, `name` and `version` by PyTorch-Lightning

    def __init__(self, conf: ConfCkptLog) -> None:

        path_ckpt = os.path.join(
            conf.dir_root,
            conf.name_exp,
            conf.name_version,
            "checkpoints",
            conf.name_ckpt
        )

        # PL's Trainer
        self.default_root_dir = conf.dir_root
        exists = get_filesystem(path_ckpt).exists(path_ckpt)
        self.resume_from_checkpoint = path_ckpt if exists else None

        # TB's TensorBoardLogger
        self.save_dir = conf.dir_root
        self.name = conf.name_exp
        self.version = conf.name_version


def generate_state_components(conf: ConfCkptLog):
    """Generate PyTorch-Lightning components for state handling.

    The components enable
      - Latest checkpointing
      - Training resume
      - Tensorboard logging

    Returns:
        (argument_dict, callback_list)
    """
    # Once 'Dictionary literal type' or something is introduced, we can handle safely.
    # {
    #     "default_root_dir": str,
    #     "resume_from_checkpoint": str,
    #     "logger": Logger Class
    # }

    ckpt_and_logging = CheckpointAndLogging(conf)

    # Save checkpoint as `last.ckpt` every 15 minutes.
    ckpt_cb = ModelCheckpoint(
        train_time_interval=timedelta(minutes=15),
        save_last=True,
        save_top_k=0,
    )

    arg_dict = {
        "default_root_dir": ckpt_and_logging.default_root_dir,
        "resume_from_checkpoint": ckpt_and_logging.resume_from_checkpoint,
        "logger": pl_loggers.TensorBoardLogger(
            ckpt_and_logging.save_dir, ckpt_and_logging.name, ckpt_and_logging.version
        )
    }
    return arg_dict, [ckpt_cb]
