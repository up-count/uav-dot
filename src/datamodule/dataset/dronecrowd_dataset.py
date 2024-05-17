from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from scipy import io

from src.datamodule.dataset.generic_dataset import GenericDataset


class DronecrowdDataset(GenericDataset):
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
        annotation_path = image_path.replace('images', 'ground_truth').replace(
            "img", "GT_img").replace('.jpg', '.mat')
        
        frame = cv2.cvtColor(cv2.imread(
            image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        labels = io.loadmat(str(annotation_path))['image_info']
        
        shape = labels.shape
        if len(shape) == 2 and shape[0] == 0:
            labels = np.zeros((0, 2), dtype=np.int64)
        elif len(shape) == 2 and shape[0] > 0 and shape[1] == 2:
            labels = labels.astype(np.int64)
        else:
            labels = labels[0, 0][0, 0][0][:, :2].astype(np.int64)
        
        labels[labels < 0] = 0
        labels[:, 0][labels[:, 0] > w - 1] = w - 1
        labels[:, 1][labels[:, 1] > h - 1] = h - 1

        # class, labels, sizes
        dest_labels = np.zeros((labels.shape[0], 5), dtype=np.float32)
        dest_labels[:, 0] = 0
        dest_labels[:, 1:3] = labels
        dest_labels[:, 3:5] = 1

        return frame, np.array(dest_labels, dtype=np.int32), np.array([0.], dtype=np.float32)
