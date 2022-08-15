import os
from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import channels_to_last


class ImagenetDataModule(pl.LightningDataModule):
    """
    Args:
        data_dir: path to the imagenet dataset file
        meta_dir: path to meta.bin file
        image_size: final image size
        num_workers: how many data workers
        batch_size: batch_size
    """

    def __init__(self,
                 data_dir: Path,
                 image_size: int = 64,
                 num_workers: int = 2,
                 batch_size: int = 32,
                 pin_memory: Optional[bool] = None,
                 setup_validation: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = True if (torch.cuda.is_available() and pin_memory is None) else False
        self.setup_validation = setup_validation

    def setup_val(self):
        val_img_dir = self.data_dir / 'val' / 'images'
        # Open and read val annotations text file
        fp = open(self.data_dir / 'val' / 'val_annotations.txt', 'r')
        data = fp.readlines()
        # Create dictionary to store img filename (word 0) and corresponding
        # label (word 1) for every line in the txt file (as key value pair)
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()
        # Create subfolders (if not present) for validation images based on label,
        # and move images into the respective folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(val_img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    def train_dataloader(self):
        transforms = self.training_transform() # if self.train_transforms is None else self.train_transforms
        train_data = datasets.ImageFolder(os.path.join(self.data_dir, 'train') , transform=transforms)
        train_loader = DataLoader(train_data,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory = self.pin_memory,
                                  )
        return train_loader

    def val_dataloader(self):
        if self.setup_validation:
            self.setup_val()
        transforms = self.validation_transform() #if self.val_transforms is None else self.val_transforms
        val_data = datasets.ImageFolder(os.path.join(self.data_dir, 'val', 'images') , transform=transforms)
        val_loader = DataLoader(val_data,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory = self.pin_memory,
                                )
        return val_loader

    def training_transform(self) -> Callable:
        """The standard imagenet transforms.
        .. code-block:: python
            transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = transforms.Compose([
                #transforms.RandomResizedCrop(self.image_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                channels_to_last,
            ])

        return preprocessing

    def validation_transform(self) -> Callable:
        """The standard imagenet transforms for validation.
        .. code-block:: python
            transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        preprocessing = transforms.Compose([
                #transforms.RandomResizedCrop(self.image_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                channels_to_last,
            ])
        return preprocessing
