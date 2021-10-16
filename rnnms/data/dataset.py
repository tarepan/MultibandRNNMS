from pathlib import Path
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass

from torch import Tensor, load
from torch.utils.data import Dataset
from corpuspy.components.archive import hash_args, try_to_acquire_archive_contents, save_archive
from omegaconf import MISSING

from .ljspeech import ConfCorpus, ItemIdLJSpeech, LJSpeech, Subtype
from .preprocess import ConfPreprocessing, preprocess_mel_mulaw


def get_dataset_mulaw_path(dir_dataset: Path, id: ItemIdLJSpeech) -> Path:
    return dir_dataset / f"{id.subtype}" / "mulaws" / f"{id.serial_num}.mulaw.pt"


def get_dataset_mel_path(dir_dataset: Path, id: ItemIdLJSpeech) -> Path:
    return dir_dataset / f"{id.subtype}" / "mels" / f"{id.serial_num}.mel.pt"


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
    corpus: ConfCorpus = ConfCorpus(mirror_root="${..adress_data_root}")
    preprocess: ConfPreprocessing = ConfPreprocessing(stft_hop_length="${..mel_stft_stride}")

class LJSpeech_mel_mulaw(Dataset):
    """Audio mel-spec/mu-law-wave dataset from LJSpeech speech corpus.
    """
    def __init__(self, train: bool, conf: ConfDataset, subtypes: List[Subtype] = list(range(1,51))):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            conf: Configuration of this dataset.
            subtypes: Sub corpus types.
        """

        # Design Notes:
        #   Download:
        #     Dataset is often saved in the private adress, so there is no `download_dataset` safety flag.
        #     `download` is common option in torchAudio datasets.
        #   Dataset archive name:
        #     Dataset contents differ based on argument, so archive should differ when arguments differ.
        #     It is guaranteed by name by argument hash.

        # Store parameters.
        self.conf = conf
        self._train = train

        self._corpus = LJSpeech(conf.corpus)
        arg_hash = hash_args(subtypes, conf.preprocess.target_sr)

        # dataset_dir_adress: URL/localPath of JSSS_spec dataset directory.
        LJS_mel_mulaw_root = Path(".")/"tmp"/"LJSpeech_mel_mulaw"
        self._path_contents_local = LJS_mel_mulaw_root/"contents"/arg_hash
        adress_dataset_dir = f"{conf.adress_data_root}/datasets/LJSpeech" if conf.adress_data_root else str(LJS_mel_mulaw_root/"archive")
        adress_dataset_archive = f"{adress_dataset_dir}/{arg_hash}.zip"

        # Prepare data identities.
        self._ids: List[ItemIdLJSpeech] = list(filter(lambda id: id.subtype in subtypes, self._corpus.get_identities()))

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(adress_dataset_archive, self._path_contents_local)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents_local, adress_dataset_archive)
            print("Dataset contents was generated and archive was saved.")

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()
        print("Preprocessing...")
        for id in self._ids:
            path_i_wav = self._corpus.get_item_path(id)
            path_o_mulaw = get_dataset_mulaw_path(self._path_contents_local, id)
            path_o_mel = get_dataset_mel_path(self._path_contents_local, id)
            preprocess_mel_mulaw(path_i_wav, path_o_mel, path_o_mulaw, self.conf.preprocess)
        print("Preprocessed.")

    def _load_datum(self, id: ItemIdLJSpeech) -> Tuple[Tensor, Tensor]:

        # Tensor(T_mel, freq)
        mel: Tensor = load(get_dataset_mel_path(self._path_contents_local, id))
        # Tensor(T_mel * hop_length,)
        mulaw: Tensor = load(get_dataset_mulaw_path(self._path_contents_local, id))

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
