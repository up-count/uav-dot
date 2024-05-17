import numpy as np
import random
import torch
from typing import Tuple
from pathlib import Path
from torch.utils.data import Dataset

from src.datamodule.dataset.mask_generator import MaskGeneratorV2


class GenericDataset(Dataset):

    def __init__(self, root_data_path: Path, image_subdir: str, image_size: Tuple[int, int], mask_size: Tuple[int, int],
                 mosaic: bool, transforms=None, normalize=None, is_test=False):

        self.root_data_path = root_data_path
        self.image_subdir = image_subdir
        self.transforms = transforms
        self.normalize = normalize
        self.image_size = image_size
        self.mask_size = mask_size
        self.mosaic = mosaic
        self.is_test = is_test

    def create_mask_generator(self, num_classes: int):
        return MaskGeneratorV2(num_classes, self.image_size, self.mask_size)

    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() > (1.0 - self.mosaic):
            image, annos = self.perform_mosaic(index)
            alt = np.array([0.])
        else:
            image, annos, alt = self.perform_normal(index)
            
        if self.is_test:
            raw_shape = image.shape[:2]
            path_name = self.images_list[index].name

        if self.transforms:
            transformed = self.transforms(
                image=image,
                keypoints=annos[:, 1:3],
                labels=annos[:, 0],
                shapes=annos[:, 3:5],
            )

            image = transformed['image']

            annos = np.zeros(
                (len(transformed['keypoints']), 5), dtype=np.int32)
            if annos.shape[0] > 0:
                annos[:, 1:3] = np.array(
                    transformed['keypoints'], dtype=np.int32)
                annos[:, 0] = np.array(transformed['labels'], dtype=np.int32)
                annos[:, 3:5] = np.array(transformed['shapes'], dtype=np.int32)
            
        mask = self.mask_generate.generate_mask(keypoints=annos)

        if self.normalize:
            image = self.normalize(image=image)['image']

        if self.is_test:
            return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask), torch.from_numpy(annos), torch.from_numpy(alt), raw_shape, path_name
        else:
            return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask), torch.from_numpy(annos), torch.from_numpy(alt)

    def perform_normal(self, index: int):
        image, annos, alt = self._load_data(index)

        return image, annos, alt

    def perform_mosaic(self, index: int):
        a = index
        b, c, d = np.random.choice(len(self), 3, replace=False)

        a_image, a_annos, _ = self._load_data(a)
        b_image, b_annos, _ = self._load_data(b)
        c_image, c_annos, _ = self._load_data(c)
        d_image, d_annos, _ = self._load_data(d)
        
        a_h, a_w = a_image.shape[:2]

        x_offset = np.random.randint(int(a_w*0.3), int(a_w*0.7))
        y_offset = np.random.randint(int(a_h*0.3), int(a_h*0.7))

        a_image, a_annos = self._filter_out_of_image(a_image, a_annos, x_offset, y_offset)
        b_image, b_annos = self._filter_out_of_image(b_image, b_annos, x_offset, a_h-y_offset)
        c_image, c_annos = self._filter_out_of_image(c_image, c_annos, a_w-x_offset, y_offset)
        d_image, d_annos = self._filter_out_of_image(d_image, d_annos, a_w-x_offset, a_h-y_offset)

        output_image = np.zeros((a_h, a_w, 3), dtype=np.uint8)
        self._insert_image_to_roi(output_image[:y_offset, :x_offset], a_image)
        self._insert_image_to_roi(output_image[y_offset:, :x_offset], b_image)
        self._insert_image_to_roi(output_image[:y_offset, x_offset:], c_image)
        self._insert_image_to_roi(output_image[y_offset:, x_offset:], d_image)

        output_annos = np.concatenate([
            a_annos, 
            b_annos + np.array([0, 0, y_offset, 0, 0]),
            c_annos + np.array([0, x_offset, 0, 0, 0]),
            d_annos + np.array([0, x_offset, y_offset, 0, 0])
            ], axis=0)

        # sort annos by left-top point
        output_annos = output_annos[np.argsort(output_annos[:, 1] + output_annos[:, 2])]

        return output_image, output_annos

    @staticmethod
    def _insert_image_to_roi(image, roi):
        h, w = roi.shape[:2]
        image[:h, :w] = roi

    @staticmethod
    def _filter_out_of_image(image, annos, x_size, y_size):
        h, w = image.shape[:2]
        
        x_size = min(max(x_size, 0), w)
        y_size = min(max(y_size, 0), h)
        
        x = np.random.randint(0, max(1, w - x_size))
        y = np.random.randint(0, max(1, h - y_size))

        image = image[y:y+y_size, x:x+x_size]
        annos = annos[np.where((annos[:, 1] > x) & (annos[:, 1] < x+x_size) & (annos[:, 2] > y) & (annos[:, 2] < y+y_size))]

        annos += np.array([0, -x, -y, 0, 0])

        return image, annos 

    def collate_fn(self, batch):
        images, masks, annos, alt = zip(*batch)

        images = torch.stack(images)
        masks = torch.stack(masks)

        return images, masks, annos, alt
    
    def collate_test_fn(self, batch):
        images, masks, annos, alt, shapes, paths = zip(*batch)

        images = torch.stack(images)
        masks = torch.stack(masks)

        return images, masks, annos, alt, shapes, paths
