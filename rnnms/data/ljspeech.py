from typing import List, NamedTuple, Optional
from pathlib import Path

from corpuspy.interface import AbstractCorpus
from corpuspy.helper.contents import get_contents


# Mode = Literal[Longform, Shortform, "simplification", "summarization"] # >=Python3.8
Subtype = int
subtypes = list(range(1,51))


class ItemIdLJSpeech(NamedTuple):
    """Identity of JSUT corpus's item.
    """
    
    subtype: Subtype
    serial_num: int


class LJSpeech(AbstractCorpus[ItemIdLJSpeech]):
    """LJSpeech corpus.
    
    Archive/contents handler of LJSpeech corpus.
    """
    
    def __init__(self, adress: Optional[str] = None, download_origin: bool = False) -> None:
        """Initiate LJSpeech with archive options.
        Args:
            adress: Corpus archive adress (e.g. path, S3) from/to which archive will be read/written through `fsspec`.
            download_origin: Download original corpus when there is no corpus in local and specified adress.
        """

        ver: str = "1.1"
        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"LJSpeech-{ver}"
        self._origin_adress = f"https://data.keithito.com/data/speech/{self._corpus_name}.tar.bz2"

        dir_corpus_local: str = "./data/corpuses/LJSpeech/"
        default_path_archive = str((Path(dir_corpus_local) / "archive" / f"{self._corpus_name}.tar.bz2").resolve())
        self._path_contents_local = Path(dir_corpus_local) / "contents"
        self._adress = adress if adress else default_path_archive

        self._download_origin = download_origin

    def get_contents(self) -> None:
        """Get corpus contents into local.
        """

        get_contents(self._adress, self._path_contents_local, self._download_origin, self.forward_from_origin)

    def forward_from_origin(self) -> None:
        """Forward original corpus archive to the adress.
        """

        forward_from_general(self._origin_adress, self._adress)

    def get_identities(self) -> List[ItemIdLJSpeech]:
        """Get corpus item identities.
        Returns:
            Full item identity list.
        """

        maxes: List[int] = [186, 338, 349, 250, 300, 308, 243, 319, 304, 317, 293, 296, 268, 340, 314, 446, 284, 398, 399, 108, 210,
            203, 141, 143, 176, 166, 180, 519, 213, 255, 233, 275, 214, 219, 210, 218, 269, 306, 248, 240, 203, 251, 188,
            239, 250, 254, 250, 289, 230, 278]
        subtype_info = {i+1: list(range(1, num+1)) for i, num in enumerate(maxes)}

        # patch
        # [index, missings]
        missings = [(2, [115]), (3, [272]), (4, [53]), (5, [81]), (6, [37]), (8, [179]), (14, [145, 270, 284, 319]),
            (16, [83, 269, 270, 345, 372, 437]), (17, [275, 279]), (21, [13]), (27, [140]), (28, [135]), (34, [139]),
            (38, [195, 196]), (42, [34, 243]), (44, [46, 216]), (48, [108]), (49, [131])]
        for msg in missings:
            subtype_info[msg[0]] = list(filter(lambda i: i not in msg[1], subtype_info[msg[0]]))


        ids: List[ItemIdLJSpeech] = []
        for subtype in subtypes:
            for num in subtype_info[subtype]:
                ids.append(ItemIdLJSpeech(subtype, num))
        return ids

    def get_item_path(self, id: ItemIdLJSpeech) -> Path:
        """Get path of the item.
        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """

        root = str(self._path_contents_local)
        group = str(id.serial_num).zfill(3)
        num = str(id.serial_num).zfill(4)
        p = f"{root}/{self._corpus_name}/wavs/LJ{group}-{num}.wav"
        return Path(p)


import fsspec


def forward_from_general(adress_from: str, forward_to: str) -> None:
    """Forward a file from the adress to specified adress.
    Forward any_adress -> any_adress through fsspec (e.g. local, S3, GCP).
    Args:
        adress_from: Forward origin adress.
        forward_to: Forward distination adress.
    """

    adress_from_with_cache = f"simplecache::{adress_from}" 
    forward_to_with_cache = f"simplecache::{forward_to}"

    with fsspec.open(adress_from_with_cache, "rb") as origin:
        print("Forward: Reading from the adress...")
        archive = origin.read()
        print("Forward: Read.")

        print("Forward: Writing to the adress...")
        with fsspec.open(forward_to_with_cache, "wb") as destination:
            destination.write(archive)
        print("Forward: Written.")
