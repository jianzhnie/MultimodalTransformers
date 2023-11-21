import os
from typing import Any, Optional

import albumentations as A
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class Flikr8kDataset(Dataset):

    def __init__(
        self,
        image_dir: str,
        tokenizer: Optional[str] = None,
        target_size: Optional[int] = None,
        max_length: int = 256,
    ) -> None:
        """image_filenames and cpations must have the same length; so, if there
        are multiple captions for each image, the image_filenames must have
        repetitive file names."""

        self.target_size = target_size
        self.image_filenames, self.captions = self.fetch_dataset(image_dir)
        self.encoded_captions = tokenizer(
            list(self.captions),
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        self.transforms = A.Compose([
            A.Resize(target_size, target_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])

    def fetch_dataset(self, data_dir: str) -> tuple[list[str], list]:
        annotations = pd.read_csv(os.path.join(data_dir, 'captions.csv'))
        image_files = [
            os.path.join(data_dir, 'Images', image_file)
            for image_file in annotations['image'].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations['caption'].to_list()
        return image_files, captions

    def __getitem__(self, idx) -> dict[str, Any]:
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(self.image_filenames[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']

        inputs = {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'images': torch.tensor(image).permute(2, 0, 1).float(),
            'caption': self.captions[idx],
        }
        return inputs

    def __len__(self) -> int:
        return len(self.captions)
