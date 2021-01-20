from argparse import ArgumentParser

import pytorch_lightning
import torchaudio

from rnnms.args import parseArgments
from rnnms.data.datamodule import DataLoaderPerformance, LJSpeechDataModule
from rnnms.train import train


def main_train():
    """Train rnnms with cli arguments and the default dataset.
    """

    torchaudio.set_audio_backend("sox_io")

    # Hardcoded hyperparams
    batch_size = 32
    seed = 1234

    # Random seed
    pytorch_lightning.seed_everything(seed)

    # Args
    parser = ArgumentParser()
    args_scpt = parseArgments(parser)

    # Datamodule
    loader_perf = DataLoaderPerformance(args_scpt.num_workers, not args_scpt.no_pin_memory)
    datamodule = LJSpeechDataModule(batch_size, performance=loader_perf, adress_data_root=args_scpt.adress_data_root)

    # Train
    train(args_scpt, datamodule)


if __name__ == "__main__":  # pragma: no cover
    main_train()