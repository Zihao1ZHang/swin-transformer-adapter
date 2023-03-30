from os.path import join
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image, ImageOps

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import list_dir, list_files


class Omniglot(VisionDataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """

    folder = "omniglot-py"
    download_url_prefix = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"
    zips_md5 = {
        "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
    }

    def __init__(
        self,
        root: str,
        background: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(join(root, self.folder), transform=transform, target_transform=target_transform)
        self.background = background

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters: List[str] = sum(
            ([join(a, c) for c in list_dir(join(self.target_folder, a))] for a in self._alphabets), []
        )
        self._character_images = [
            [(image, idx) for image in list_files(join(self.target_folder, character), ".png")]
            for idx, character in enumerate(self._characters)
        ]
        self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])

    def __len__(self) -> int:
        return len(self._flat_character_images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode="r").convert("L")
        # image = ImageOps.colorize(image, black='black', white='white')
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _get_target_folder(self) -> str:
        return "images_background" if self.background else "images_evaluation"