from pathlib import Path
import random
from typing import List, Optional, Tuple

from torch import Tensor, load
from torch.utils.data import Dataset
from corpuspy.components.archive import hash_args, try_to_acquire_archive_contents, save_archive

from .ljspeech import ItemIdLJSpeech, LJSpeech, Subtype
from .preprocess import preprocess_mel_mulaw


def get_dataset_mulaw_path(dir_dataset: Path, id: ItemIdLJSpeech) -> Path:
    return dir_dataset / f"{id.subtype}" / "mulaws" / f"{id.serial_num}.mulaw.pt"


def get_dataset_mel_path(dir_dataset: Path, id: ItemIdLJSpeech) -> Path:
    return dir_dataset / f"{id.subtype}" / "mels" / f"{id.serial_num}.mel.pt"


class LJSpeech_mel_mulaw(Dataset):
    """Audio mel-spec/mu-law-wave dataset from LJSpeech speech corpus.
    """
    def __init__(
        self,
        train: bool,
        subtypes: List[Subtype] = list(range(1,51)),
        download_corpus: bool = False,
        corpus_adress: Optional[str] = None,
        dataset_dir_adress: Optional[str] = None,
    ):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            subtypes: Sub corpus types.
            download_corpus: Whether download the corpus or not when dataset is not found.
            corpus_adress: URL/localPath of corpus archive (remote url, like `s3::`, can be used). None use default URL.
            dataset_dir_adress: URL/localPath of JSSS_spec dataset directory (remote url, like `s3::`, can be used).
        """

        # Design Notes:
        #   Sampling rate:
        #     Sampling rates of dataset A and B should match, so `sampling_rate` is not a optional, but required argument.
        #   Download:
        #     Dataset is often saved in the private adress, so there is no `download_dataset` safety flag.
        #     `download` is common option in torchAudio datasets.
        #   Dataset archive name:
        #     Dataset contents differ based on argument, so archive should differ when arguments differ.
        #     It is guaranteed by name by argument hash.

        # Hardcoded Hyperparams

        # resample_sr: If not None, resample with specified sampling rate.
        # clip_length_mel: Clipping length with mel frame unit.
        # mel_stft_stride: hop length of mel-spectrogram STFT.
        resample_sr: Optional[int] = 16000
        clip_length_mel: int = 24
        mel_stft_stride: int = 200

        # Store parameters.
        self._train = train
        self._resample_sr = resample_sr
        self._clip_length_mel = clip_length_mel
        self._mel_stft_hop_length = mel_stft_stride

        self._corpus = LJSpeech(corpus_adress, download_corpus)
        arg_hash = hash_args(subtypes, resample_sr)
        LJS_mel_mulaw_root = Path(".")/"tmp"/"LJSpeech_mel_mulaw"
        self._path_contents_local = LJS_mel_mulaw_root/"contents"/arg_hash
        dataset_dir_adress = dataset_dir_adress if dataset_dir_adress else str(LJS_mel_mulaw_root/"archive")
        dataset_archive_adress = f"{dataset_dir_adress}/{arg_hash}.zip"

        # Prepare data identities.
        self._ids: List[ItemIdLJSpeech] = list(filter(lambda id: id.subtype in subtypes, self._corpus.get_identities()))

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(dataset_archive_adress, self._path_contents_local)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents_local, dataset_archive_adress)
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
            preprocess_mel_mulaw(path_i_wav, path_o_mel, path_o_mulaw, self._resample_sr, self._mel_stft_hop_length)
        print("Preprocessed.")

    def _load_datum(self, id: ItemIdLJSpeech) -> Tuple[Tensor, Tensor]:

        # Tensor(T_mel, freq)
        mel: Tensor = load(get_dataset_mel_path(self._path_contents_local, id))
        # Tensor(T_mel * hop_length,)
        mulaw: Tensor = load(get_dataset_mulaw_path(self._path_contents_local, id))

        # Random clipping
        start = random.randint(0, mel.size()[-1] - self._clip_length_mel - 1)

        # Mel-spectrogram clipping
        start_mel = start
        end_mel = start + self._clip_length_mel
        # (T_mel, freq) -> (clip_length_mel, freq)
        mel_clipped = mel[:, start_mel : end_mel]

        # Waveform clipping
        start_mulaw = self._mel_stft_hop_length * start_mel
        end_mulaw = self._mel_stft_hop_length * end_mel + 1
        # (T_mel * hop_length,) -> (clip_length_mel * hop_length,)
        mulaw_clipped = mulaw[start_mulaw : end_mulaw]

        return (mulaw_clipped, mel_clipped)

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset.
        Args:
            n : The index of the datum to be loaded
        """
        return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)