from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torchaudio
from omegaconf import MISSING

from rnnms.data.datamodule import DataLoaderPerformance, LJSpeechDataModule
from rnnms.train import train, ConfTrain
from rnnms.config import load_conf


@dataclass
class ConfData:
    """Configuration of data.
    Args:
        batch_size: Number of datum in a batch
        num_workers: Number of data loader worker
        pin_memory: Use data loader pin_memory
        adress_data_root: Root adress of data
    """
    batch_size: int = MISSING
    num_workers: Optional[int] = MISSING
    pin_memory: Optional[bool] = MISSING
    adress_data_root: Optional[str] = MISSING

@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        seed: PyTorch-Lightning's seed for every random system
        path_extend_conf: Path of configuration yaml which extends default config
    """
    seed: int = MISSING
    path_extend_conf: Optional[str] = MISSING
    data: ConfData = ConfData()
    train: ConfTrain = ConfTrain()

def main_train():
    """Train rnnms with cli arguments and the default dataset.
    """

    # Load default/extend/CLI configs.
    conf = load_conf()

    # Setup
    pl.seed_everything(conf.seed)
    torchaudio.set_audio_backend("sox_io")

    # Dataset
    datamodule = LJSpeechDataModule(
        conf.data.batch_size,
        performance=DataLoaderPerformance(conf.data.num_workers, conf.data.pin_memory),
        adress_data_root=conf.data.adress_data_root
    )

    # Train
    train(conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()
