from dataclasses import dataclass

import pytorch_lightning as pl
import torchaudio
from omegaconf import MISSING

from rnnms.data.datamodule import DataLoaderPerformance, LJSpeechDataModule
from rnnms.train import train, ConfTrain



@dataclass
class ConfData:
    batch_size: int = 32
    num_workers: int = MISSING
    pin_memory: bool = MISSING
    adress_data_root: str = MISSING

@dataclass
class ConfGlobal:
    seed = 1234
    data: ConfData = ConfData()
    train: ConfTrain = ConfTrain()

def main_train():
    """Train rnnms with cli arguments and the default dataset.
    """

    conf = ConfGlobal()

    # Setup
    pl.seed_everything(conf.seed)
    torchaudio.set_audio_backend("sox_io")

    # Datamodule
    loader_perf = DataLoaderPerformance(conf.data.num_workers, conf.data.pin_memory)
    datamodule = LJSpeechDataModule(conf.data.batch_size, performance=loader_perf, adress_data_root=conf.data.adress_data_root)

    # Train
    train(conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()