"""Run RNNMS training"""


import pytorch_lightning as pl
import torchaudio

from mbrnnms.data.datamodule import generate_datamodule
from mbrnnms.train import train
from mbrnnms.config import load_conf


def main_train():
    """Train rnnms with cli arguments and the default dataset.
    """

    # Load default/extend/CLI configs.
    conf = load_conf()

    # Setup
    pl.seed_everything(conf.seed)
    torchaudio.set_audio_backend("sox_io")

    # Dataset
    datamodule = generate_datamodule(conf.data)

    # Train
    train(conf.train, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()
