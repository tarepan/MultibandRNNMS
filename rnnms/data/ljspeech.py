from typing import List, NamedTuple, Optional
from pathlib import Path
from dataclasses import dataclass

from corpuspy.interface import AbstractCorpus
from corpuspy.helper.contents import get_contents
from omegaconf import MISSING
import fsspec

# Mode = Literal[Longform, Shortform, "simplification", "summarization"] # >=Python3.8
Subtype = int
subtypes = list(range(1,51))


class ItemIdLJSpeech(NamedTuple):
    """Identity of JSUT corpus's item.
    """

    subtype: Subtype
    serial_num: int


@dataclass
class ConfCorpus:
    """Configuration of corpus.

    Args:
        mirror_root: Root adress of corpus mirror, to which original archive is forwarded. If None, use default.
        download: Whether download original corpus or not when requested (e.g. origin->mirror forwarding).
    """
    mirror_root: Optional[str] = MISSING
    download: bool = MISSING

class LJSpeech(AbstractCorpus[ItemIdLJSpeech]):
    """LJSpeech corpus.

    Archive/contents handler of LJSpeech corpus.

    Terminology:
        mirror: Mirror archive of the corpus
        contents: Contents extracted from archive
    """

    def __init__(self, conf: ConfCorpus) -> None:
        """Initiate LJSpeech with archive options.
        """

        self.conf = conf

        ver: str = "1.1"
        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"LJSpeech-{ver}"
        archive_name = f"{self._corpus_name}.tar.bz2"

        self._origin_adress = f"https://data.keithito.com/data/speech/{archive_name}"

        mirror_root = conf.mirror_root
        # Directory to which contents are extracted, and mirror is placed if adress is not provided.
        local_root = Path("./data")

        # Mirror: placed in given adress (conf) or default adress (local corpus directory)
        adress_mirror_given = f"{mirror_root}/corpuses/{archive_name}" if mirror_root else None
        adress_mirror_default = str((local_root / "corpuses" / "LJSpeech" / "archive" / archive_name).resolve())
        self._adress_mirror = adress_mirror_given or adress_mirror_default

        # Contents: contents are extracted in local corpus directory
        self._path_contents = local_root / "corpuses" / "LJSpeech" / "contents"

    def get_contents(self) -> None:
        """Get corpus contents into local.
        """

        get_contents(self._adress_mirror, self._path_contents, self.conf.download, self.forward_from_origin)

    def forward_from_origin(self) -> None:
        """Forward original corpus archive to the mirror adress.
        """

        forward_from_general(self._origin_adress, self._adress_mirror)

    def get_identities(self) -> List[ItemIdLJSpeech]:
        """Get corpus item identities.
        Returns:
            Full item identity list.
        """

        # Design notes:
        #   No contents dependency is intentional.
        #   Corpus handler can be used without corpus itself (e.g. Get item identities for a preprocessed dataset).
        #   Hard-coded identity list enable contents-independent identity acquisition.

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

        root = self._path_contents
        subtype = str(id.subtype).zfill(3)
        num = str(id.serial_num).zfill(4)
        return root / self._corpus_name / "wavs" / f"LJ{subtype}-{num}.wav"


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
