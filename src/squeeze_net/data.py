"""Get data and define datasets."""

from enum import StrEnum
from datasets import load_dataset
from torch.utils.data import DataLoader


class Split(StrEnum):
    """Describes what type of split to use in the dataloader"""

    TRAIN = "train"
    TEST = "test"
    VAL = "validation"


class ImageNetDataLoader(DataLoader):
    """Create an ImageNetDataloader"""

    def __init__(self, batch_size: int = 4, split: Split = Split.TRAIN):
        # Streaming feature allows me to not download the entire dataset locally.
        # https://stackoverflow.com/questions/75481137/is-there-is-a-way-that-i-can-download-only-a-part-of-the-dataset-from-huggingfac
        dataset = load_dataset(
            "ILSVRC/imagenet-1k", split=split, trust_remote_code=True, streaming=True
        )
        dataset.with_format("torch")
        super().__init__(dataset=dataset, batch_size=batch_size)
