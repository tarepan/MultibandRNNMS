"""Datasets"""


import random
from typing import List, Optional, Tuple
from dataclasses import dataclass

from torch import Tensor, load
from torch.utils.data import Dataset
from tqdm import tqdm
from omegaconf import MISSING

from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId
from speechdatasety.helper.archive import hash_args
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive
from speechdatasety.helper.adress import dataset_adress, generate_path_getter

from mbrnnms.networks.pqmf import PQMF

from .preprocess import ConfPreprocessing, preprocess_mel_mb_mulaw


@dataclass
class ConfDataset:
    """Configuration of dataset.

    Args:
        adress_data_root: Root adress of data
        clip_length_mel: Clipping length with mel frame unit.
        mel_stft_stride: hop length of mel-spectrogram STFT.
    """
    adress_data_root: Optional[str] = MISSING
    clip_length_mel: int = MISSING
    mel_stft_stride: int = MISSING
    n_band: int = MISSING
    preprocess: ConfPreprocessing = ConfPreprocessing(stft_hop_length="${..mel_stft_stride}")

class MelMulaw(Dataset):
    """Audio mel-spec/mu-law-wave dataset from the corpus.
    """
    def __init__(
        self,
        train: bool,
        conf: ConfDataset,
        corpus: AbstractCorpus,
    ):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            conf: Configuration of this dataset.
            corpus: Corpus instance
        """

        # Store parameters.
        self.conf = conf
        self._train = train
        self._corpus = corpus
        arg_hash = hash_args(
            conf.preprocess.bits_mulaw,
            conf.preprocess.stft_hop_length,
            conf.preprocess.target_sr,
        )

        adress_archive, self._path_contents = dataset_adress(
            conf.adress_data_root,
            corpus.__class__.__name__,
            "mel_mulaw_mb",
            arg_hash,
        )

        # Prepare datum path getter.
        self.get_path_mel = generate_path_getter("mel", self._path_contents)
        self.get_path_mulaw = generate_path_getter("mulaw_mb", self._path_contents)

        # Prepare data identities.
        self._ids: List[ItemId] = self._corpus.get_identities()

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_archive, self._path_contents)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents, adress_archive)
            print("Dataset contents was generated and archive was saved.")

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()
        pqmf = PQMF()
        for item_id in tqdm(self._ids, desc="Preprocessing", unit="utterance"):
            preprocess_mel_mb_mulaw(
                self._corpus.get_item_path(item_id),
                self.get_path_mel(item_id),
                self.get_path_mulaw(item_id),
                pqmf,
                self.conf.preprocess
            )

    def _load_datum(self, item_id: ItemId) -> Tuple[Tensor, Tensor]:

        # Mel-spectrogram :: Tensor(T_mel, freq)
        mel: Tensor = load(self.get_path_mel(item_id))
        # Multiband μ-law :: Tensor(Band, T_mel * hop_length // n_band,)
        mb_mulaw: Tensor = load(self.get_path_mulaw(item_id))

        if self._train:
            # Time-directional random clipping
            start = random.randint(0, mel.size()[-2] - self.conf.clip_length_mel - 1)

            # Mel-spectrogram clipping
            start_mel = start
            end_mel = start + self.conf.clip_length_mel
            # (T_mel, freq) -> (clip_length_mel, freq)
            mel_clipped = mel[start_mel : end_mel]

            # Waveform clipping
            stride = self.conf.mel_stft_stride // self.conf.n_band
            start_mb_mulaw = stride * start_mel
            end_mb_mulaw = stride * end_mel + 1
            # (Band, T_mel * stride) -> (Band, clip_length_mel * stride)
            mb_mulaw_clipped = mb_mulaw[start_mb_mulaw : end_mb_mulaw]

            return mb_mulaw_clipped, mel_clipped
        else:
            return mb_mulaw, mel

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.
        Args:
            n : The index of the datum to be loaded
        """
        return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)
