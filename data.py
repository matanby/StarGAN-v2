import os

import numpy as np
from torch.utils.data import Dataset
from typing import Dict

import image_utils
import utils


class FFHQDataset(Dataset):
    def __init__(self, dataset_path: str):
        self._prepare_paths(dataset_path)

    def __getitem__(self, idx: int) -> Dict:
        assert idx < len(self)

        image = self._load_image(idx)
        attributes = self._load_attributes(idx)

        item = {
            'image': image,
            'attributes': attributes
        }

        return item

    def _prepare_paths(self, dataset_path: str) -> None:
        self._image_paths = []
        self._attribute_paths = []

        images_root = os.path.join(dataset_path, 'images')
        jsons_root = os.path.join(dataset_path, 'face-attributes')

        image_paths = image_utils.list_images(images_root)
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image_name_no_ext = os.path.splitext(image_name)[0]
            json_name = f'{image_name_no_ext}.json'
            json_path = os.path.join(jsons_root, json_name)

            if not os.path.exists(json_path):
                continue

            attrs = utils.load_json(json_path)
            if not (isinstance(attrs, list) and len(attrs) == 1):
                continue

            self._image_paths.append(image_path)
            self._attribute_paths.append(json_path)

    def _load_image(self, idx: int) -> np.ndarray:
        image_path = self._image_paths[idx]
        image = image_utils.load(image_path, channels_first=True)
        image = (image * 2) - 1
        return image


    def _load_attributes(self, idx: int) -> Dict[str, float]:
        assert idx < len(self)

        attr_json_path = self._attribute_paths[idx]
        attrs_json = utils.load_json(attr_json_path)[0]

        attrs = {
            'gender': 0 if attrs_json['faceAttributes']['gender'] == 'female' else 1,
            'age': attrs_json['faceAttributes']['age'],
        }

        return attrs

    def __len__(self):
        return len(self._image_paths)
