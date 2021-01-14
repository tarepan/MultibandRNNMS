from typing import Optional
from os import cpu_count

from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import LJSpeech_mel_mulaw


class DataLoaderPerformance:
    """PyTorch DataLoader performance configs.
    All attributes which affect performance of [torch.utils.data.DataLoader][^DataLoader] @ v1.6.0
    [^DataLoader]:https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    def __init__(self, num_workers: Optional[int] = None, pin_memory: bool = True) -> None:
        """Default: num_workers == cpu_count & pin_memory == True
        """

        # Design Note:
        #   Current performance is for single GPU training.
        #   cpu_count() is not appropriate under the multi-GPU condition.

        if num_workers is None:
            c = cpu_count()
            num_workers = c if c is not None else 0
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory


class LJSpeechDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        performance: Optional[DataLoaderPerformance] = None,
        adress_data_root: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root = adress_data_root or "./"

        # Performance setup
        if performance is None:
            performance = DataLoaderPerformance()
        self._num_worker = performance.num_workers
        self._pin_memory = performance.pin_memory

        self._adress_corpuses = f"{adress_data_root}/corpuses/LJSpeech-1.1.tar.bz2" if adress_data_root else None
        self._adress_dir_datasets = f"{adress_data_root}/datasets" if adress_data_root else None

    def prepare_data(self, *args, **kwargs) -> None:
        LJSpeech_mel_mulaw(
            train=True,
            download_corpus=True,
            corpus_adress=self._adress_corpuses,
            dataset_dir_adress=self._adress_dir_datasets,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_full = LJSpeech_mel_mulaw(
                train=True,
                download_corpus=True,
                corpus_adress=self._adress_corpuses,
                dataset_dir_adress=self._adress_dir_datasets,
            )

            # use modulo for validation (#training become batch*N)
            n_full = len(dataset_full)
            mod = n_full % self.batch_size
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [n_full - mod, mod]
            )
            self.batch_size_val = mod
        if stage == "test" or stage is None:
            self.dataset_test = LJSpeech_mel_mulaw(
                train=False,
                download_corpus=True,
                corpus_adress=self._adress_corpuses,
                dataset_dir_adress=self._adress_dir_datasets,
            )
            self.batch_size_test = self.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )
