"""Datasets"""


from pathlib import Path
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass

from torch import Tensor, load
from torch.utils.data import Dataset
from omegaconf import MISSING
from speechcorpusy.interface import AbstractCorpus, ItemId
from speechcorpusy.components.archive import hash_args
from speechcorpusy.components.archive import try_to_acquire_archive_contents, save_archive

from .preprocess import ConfPreprocessing, preprocess_mel_mulaw


def get_dataset_mulaw_path(dir_dataset: Path, item_id: ItemId) -> Path:
    """Get waveform item path in dataset.
    """
    return dir_dataset / f"{item_id.speaker}" / "mulaws" / f"{item_id.name}.mulaw.pt"


def get_dataset_mel_path(dir_dataset: Path, item_id: ItemId) -> Path:
    """Get mel-spec item path in dataset.
    """
    return dir_dataset / f"{item_id.speaker}" / "mels" / f"{item_id.name}.mel.pt"


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
        archive_name = f"{arg_hash}.zip"

        archive_root = conf.adress_data_root
        # Directory to which contents are extracted and archive is placed
        # if adress is not provided.
        local_root = Path(".")/"tmp"/"LJSpeech_mel_mulaw"

        # Archive: placed in given adress (conf) or default adress (local dataset directory)
        adress_archive_given = f"{archive_root}/datasets/LJSpeech/{archive_name}" if archive_root else None
        adress_archive_default = str(local_root/"archive"/archive_name)
        adress_archive = adress_archive_given or adress_archive_default

        # Contents: contents are extracted in local dataset directory
        self._path_contents = local_root/"contents"/arg_hash

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
        print("Preprocessing...")
        for item_id in self._ids:
            path_i_wav = self._corpus.get_item_path(item_id)
            path_o_mulaw = get_dataset_mulaw_path(self._path_contents, item_id)
            path_o_mel = get_dataset_mel_path(self._path_contents, item_id)
            preprocess_mel_mulaw(path_i_wav, path_o_mel, path_o_mulaw, self.conf.preprocess)
        print("Preprocessed.")

    def _load_datum(self, item_id: ItemId) -> Tuple[Tensor, Tensor]:

        # Tensor(T_mel, freq)
        mel: Tensor = load(get_dataset_mel_path(self._path_contents, item_id))
        # Tensor(T_mel * hop_length,)
        mulaw: Tensor = load(get_dataset_mulaw_path(self._path_contents, item_id))

        if self._train:
            # Time-directional random clipping
            start = random.randint(0, mel.size()[-2] - self.conf.clip_length_mel - 1)

            # Mel-spectrogram clipping
            start_mel = start
            end_mel = start + self.conf.clip_length_mel
            # (T_mel, freq) -> (clip_length_mel, freq)
            mel_clipped = mel[start_mel : end_mel]

            # Waveform clipping
            start_mulaw = self.conf.mel_stft_stride * start_mel
            end_mulaw = self.conf.mel_stft_stride * end_mel + 1
            # (T_mel * hop_length,) -> (clip_length_mel * hop_length,)
            mulaw_clipped = mulaw[start_mulaw : end_mulaw]

            return mulaw_clipped, mel_clipped
        else:
            return mulaw, mel

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.
        Args:
            n : The index of the datum to be loaded
        """
        return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)
