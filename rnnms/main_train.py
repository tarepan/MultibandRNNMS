import pytorch_lightning as pl
import torchaudio

from rnnms.data.datamodule import DataLoaderPerformance, LJSpeechDataModule
from rnnms.train import train
from rnnms.config import load_conf


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
