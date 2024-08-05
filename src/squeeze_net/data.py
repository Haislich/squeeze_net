"""Get data and define datasets."""

from enum import StrEnum
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class Split(StrEnum):
    """Describes what type of split to use in the dataloader"""

    TRAIN = "train"
    TEST = "test"
    VAL = "validation"


class ImageNetDataLoader(DataLoader):
    """Create an ImageNetDataloader"""

    _preprocess_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(0.5, 0.5, False),
        ]
    )

    def __init__(self, batch_size: int = 4, split: Split = Split.TRAIN):

        # Streaming feature allows me to not download the entire dataset locally.
        # https://stackoverflow.com/questions/75481137/is-there-is-a-way-that-i-can-download-only-a-part-of-the-dataset-from-huggingface
        dataset = (
            load_dataset(
                "imagenet-1k",
                split=split,
                trust_remote_code=True,
                streaming=True,
            )
            .with_format("torch")
            .map(self._preprocess)
        )

        super().__init__(dataset=dataset, batch_size=batch_size)

    def _preprocess(self, data):
        if data["image"].shape[0] < 3:
            data["image"] = data["image"].repeat(3, 1, 1)
        data["image"] = self._preprocess_transform(data["image"].float())
        return data
