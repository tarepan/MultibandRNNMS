"""MulawMel datamodule"""


from typing import Optional
from os import cpu_count
from dataclasses import dataclass

import torch
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from omegaconf import MISSING
from speechcorpusy.interface import AbstractCorpus, ConfCorpus
from speechcorpusy import load_preset

from .dataset import ConfDataset, MBMelMulaw


@dataclass
class ConfLoader:
    """Configuration of data loader.
    Args:
        batch_size: Number of datum in a batch
        num_workers: Number of data loader worker
        pin_memory: Use data loader pin_memory
    """
    batch_size: int = MISSING
    num_workers: Optional[int] = MISSING
    pin_memory: Optional[bool] = MISSING

class DataLoaderPerformance:
    """PyTorch DataLoader performance configs.
    All attributes which affect performance of [torch.utils.data.DataLoader][^DataLoader] @ v1.6.0
    [^DataLoader]:https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    num_workers: int
    pin_memory: bool

    def __init__(self, num_workers: Optional[int], pin_memory: Optional[bool]) -> None:
        """Default: num_workers == cpu_count & pin_memory == True
        """

        # Design Note:
        #   Current performance is for single GPU training.
        #   cpu_count() is not appropriate under the multi-GPU condition.

        if num_workers is None:
            n_cpu = cpu_count()
            num_workers = n_cpu if n_cpu is not None else 0
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory if pin_memory is not None else True


@dataclass
class ConfData:
    """Configuration of data.
    """
    data_name: str = MISSING
    adress_data_root: Optional[str] = MISSING
    loader: ConfLoader = ConfLoader()
    dataset: ConfDataset = ConfDataset(
        adress_data_root="${..adress_data_root}"
    )
    corpus: ConfCorpus = ConfCorpus(
        root="${..adress_data_root}",
    )

class MelMulawDataModule(LightningDataModule):
    """PL-DataModule of mel & μ-law wave.
    """
    def __init__(self, conf: ConfData, corpus: AbstractCorpus):
        # Design Notes: Dataset independent
        #   DataModule's responsibility is about data loader.
        #   Dataset handle corpus, preprocessing and datum sampling.

        super().__init__()
        self._conf = conf
        self._corpus = corpus
        self._loader_perf = DataLoaderPerformance(conf.loader.num_workers, conf.loader.pin_memory)

    def prepare_data(self) -> None:
        """Prepare data in dataset.
        """

        MBMelMulaw(True, self._conf.dataset, self._corpus)

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets and batch sizes.
        """

        if stage == "fit" or stage is None:
            dataset_full_train = MBMelMulaw(True, self._conf.dataset, self._corpus)
            dataset_full_not_train = MBMelMulaw(False, self._conf.dataset, self._corpus)

            n_full = len(dataset_full_train)
            # N-3 utterances, fixed-length.
            self.dataset_train, _ = random_split(
                dataset_full_train, [n_full - 3, 3], generator=torch.Generator().manual_seed(42)
            )
            # 3 utterances, variable-length. Generator enable consistent split.
            _, self.dataset_val = random_split(
                dataset_full_not_train, [n_full - 3, 3], generator=torch.Generator().manual_seed(42)
            )
            self.batch_size_val = 1

        if stage == "test" or stage is None:
            self.dataset_test = MBMelMulaw(False, self._conf.dataset, self._corpus)
            self.batch_size_test = self._conf.loader.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self._conf.loader.batch_size,
            shuffle=True,
            num_workers=self._loader_perf.num_workers,
            pin_memory=self._loader_perf.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            num_workers=self._loader_perf.num_workers,
            pin_memory=self._loader_perf.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            num_workers=self._loader_perf.num_workers,
            pin_memory=self._loader_perf.pin_memory,
        )


def generate_datamodule(conf: ConfData) -> MelMulawDataModule:
    """Generate datamodule with given corpus"""
    corpus = load_preset(conf.data_name, conf=conf.corpus)
    return MelMulawDataModule(conf, corpus)
