import cv2
import os
import torch
import random
from pathlib import Path
from typing import List, Tuple
from random import Random
import itertools
from collections import deque

import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader



class DotDatamodule(LightningDataModule):
    def __init__(self,
                 root_data_path: Path,
                 dataset: str,
                 data_fold: int,
                 batch_size: int,
                 workers: int,
                 data_mean: List[float],
                 data_std: List[float],
                 image_size: Tuple[int, int],
                 mask_size: Tuple[int, int],
                 mosaic: float,
                 ) -> None:
        super().__init__()

        self.root_data_path = root_data_path
        self.dataset = dataset
        self.data_fold = data_fold
        self.workers = workers
        self.batch_size = batch_size
        self.data_mean = data_mean
        self.data_std = data_std
        self.image_size = image_size
        self.mask_size = mask_size
        self.mosaic = mosaic
        
        self.save_hyperparameters()
        
        if self.dataset == 'dronecrowd':
            from src.datamodule.dataset.dronecrowd_dataset import DronecrowdDataset as DotDataset
            train_data = sorted(Path(os.path.join(self.root_data_path, 'train_data', 'images')).glob('*.jpg'))
            val_data = sorted(Path(os.path.join(self.root_data_path, 'val_data', 'images')).glob('*.jpg'))
            test_data = sorted(Path(os.path.join(self.root_data_path, 'test_data', 'images')).glob('*.jpg'))
            
            if self.data_fold >= 0:
                splits = self.prepare_splits(train_data + val_data + test_data)
                splits = self.splits_to_files_list(splits)
                train_data, val_data, test_data = self.get_train_valid_test(splits, self.data_fold)
        
        elif self.dataset == 'upcount':
            from src.datamodule.dataset.upcount_dataset import UpcountDataset as DotDataset
            
            with open(os.path.join(self.root_data_path, 'train.txt'), 'r') as f:
                train_sequences = f.read().splitlines()
                
            with open(os.path.join(self.root_data_path, 'val.txt'), 'r') as f:
                val_sequences = f.read().splitlines()
                
            with open(os.path.join(self.root_data_path, 'test.txt'), 'r') as f:
                test_sequences = f.read().splitlines()
                
            train_data = []
            for seq in train_sequences:
                train_data += sorted(Path(os.path.join(self.root_data_path, 'images', seq)).glob('*.jpg'))
                
            val_data = []
            for seq in val_sequences:
                val_data += sorted(Path(os.path.join(self.root_data_path, 'images', seq)).glob('*.jpg'))
            
            test_data = []
            for seq in test_sequences:
                test_data += sorted(Path(os.path.join(self.root_data_path, 'images', seq)).glob('*.jpg'))
        else:
            print('Invalid dataset name. Exiting...')
            exit()
        
        if self.dataset == 'dronecrowd':
            train_data = sorted(train_data)[::4]

        augmentation = A.Compose([
            A.Flip(),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
                                    
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
            ], p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            ], p=0.5),
            A.RandomGamma((40, 220), p=0.8),
            
            A.RandomResizedCrop(height=self.image_size[1], width=self.image_size[0], scale=(0.5, 1.0), ratio=(1.0, 1.0), p=0.4),
            A.Resize(height=self.image_size[1], width=self.image_size[0], always_apply=True),
        ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, label_fields=['labels', 'shapes'])
        )
        
        transforms = A.Compose([
            A.Resize(height=self.image_size[1], width=self.image_size[0], always_apply=True),
        ], 
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=True, label_fields=['labels', 'shapes'])
        )
        
        normalize = A.Normalize(mean=self.data_mean, std=self.data_std, max_pixel_value=255.0)

        self.train_dataset = DotDataset(
            root_data_path,
            train_data,
            image_subdir='train',
            image_size=self.image_size,
            mask_size=self.mask_size,
            mosaic=self.mosaic,
            transforms=augmentation,
            normalize=normalize)

        self.val_dataset = DotDataset(
            root_data_path,
            val_data,
            image_subdir='val',
            image_size=self.image_size,
            mask_size=self.mask_size,
            mosaic=False,
            transforms=transforms,
            normalize=normalize)

        self.test_dataset = DotDataset(
            root_data_path,
            test_data,
            image_subdir='test',
            image_size=self.image_size,
            mask_size=self.mask_size,
            mosaic=False,
            transforms=transforms,
            normalize=normalize,
            is_test=True)
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=False,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=False,
            collate_fn=self.test_dataset.collate_test_fn,
        )

    def get_train_steps(self):
        return len(self.train_dataset) // self.batch_size
    
    def prepare_splits(self, files_list):
        sequences_list = []

        for sequence_name in files_list:
            idx = f'{str(sequence_name).split("/")[-1][3:6]}'

            sequences_list.append(idx)

        sequences_list = sorted(list(set(sequences_list)))

        splits = self.partition_sequences(sequences_list)
        
        return splits
    
    def splits_to_files_list(self, splits):
        files_list = []
        
        for s in splits:
            s_list = []
            
            for seq in s:
                for cat in ['train', 'val', 'test']:
                    s_list += sorted(Path(os.path.join(self.root_data_path, f'{cat}_data', 'images')).glob(f'img{seq}*.jpg'))            
            files_list.append(s_list)

        return files_list
    
    @staticmethod
    def partition_sequences(sequences: List[str], n: int = 5) -> List[List[str]]:
        sequences = sequences.copy()
        Random(42).shuffle(sequences)
        return [sequences[i::n] for i in range(n)]

    @staticmethod
    def get_train_valid_test(splits: List[List[str]], current_split: int):
        splits = deque(splits)
        splits.rotate(current_split)
        splits = list(splits)

        return list(itertools.chain.from_iterable(splits[:-2])), splits[-2], splits[-1]
