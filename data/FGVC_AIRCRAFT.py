from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from typing import Callable, Optional
from PIL import Image
from torchvision.datasets import VisionDataset


class FGVCAircraft(VisionDataset):
    """`FGVC Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    The dataset contains 10,000 images of aircraft, with 100 images for each of 100
    different aircraft model variants, most of which are airplanes.
    Aircraft models are organized in a three-levels hierarchy. The three levels, from
    finer to coarser, are:

    - ``variant``, e.g. Boeing 737-700. A variant collapses all the models that are visually
        indistinguishable into one class. The dataset comprises 100 different variants.
    - ``family``, e.g. Boeing 737. The dataset comprises 70 different families.
    - ``manufacturer``, e.g. Boeing. The dataset comprises 30 different manufacturers.

    Args:
        root (string): Root directory of the FGVC Aircraft dataset.
        split (string, optional): The dataset split, supports ``train``, ``val``,
            ``trainval`` and ``test``.
        annotation_level (str, optional): The annotation level, supports ``variant``,
            ``family`` and ``manufacturer``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        annotation_level: str = "variant",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = split
        self._annotation_level = annotation_level

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")


        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        annotation_file = os.path.join(
            self._data_path,
            "data",
            {
                "variant": "variants.txt",
                "family": "families.txt",
                "manufacturer": "manufacturers.txt",
            }[self._annotation_level],
        )
        with open(annotation_file, "r") as f:
            self.classes = [line.strip() for line in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        image_data_folder = os.path.join(self._data_path, "data", "images")
        labels_file = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self._split}.txt")

        self._image_files = []
        self._labels = []

        with open(labels_file, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                self._labels.append(self.class_to_idx[label_name])

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)