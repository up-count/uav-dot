import warnings

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np

from src.datamodule.dataset.generic_dataset import GenericDataset


class UpcountDataset(GenericDataset):
    def __init__(self, root_data_path: Path, images_list: List[str], image_subdir: str, image_size: Tuple[int, int], mask_size: Tuple[int, int],
                 mosaic: float, transforms=None, normalize=None, is_test=False):
        super().__init__(root_data_path, image_subdir,
                         image_size, mask_size, mosaic, transforms, normalize, is_test)

        self.num_classes = 1
        self.images_list = images_list

        self.mask_generate = self.create_mask_generator(self.num_classes)

    def __len__(self):
        return len(self.images_list)

    def _load_data(self, index: int):
        image_path = str(self.images_list[index])
        annotation_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        alt = float(image_path.split('/')[-1].split('__')[-1][:-4])
        
        frame = cv2.cvtColor(cv2.imread(
            image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = np.loadtxt(annotation_path, dtype=np.float32).reshape(-1, 2).astype(np.int64)
                
        labels[labels < 0] = 0
        labels[:, 0][labels[:, 0] > w - 1] = w - 1
        labels[:, 1][labels[:, 1] > h - 1] = h - 1

        # class, labels, sizes
        dest_labels = np.zeros((labels.shape[0], 5), dtype=np.float32)
        dest_labels[:, 0] = 0
        dest_labels[:, 1:3] = labels
        dest_labels[:, 3:5] = 1

        return frame, np.array(dest_labels, dtype=np.int32), np.array([alt], dtype=np.float32)
