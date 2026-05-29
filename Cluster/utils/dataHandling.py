"""Instead of loading data in the seperate files, load it centrally here, so that data handling only has to be managed in one location"""

import os
import argparse

import torch
from torchvision import datasets    # type: ignore
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import v2, Compose, Resize, ToTensor    # type: ignore

from PIL import Image

class FlatDirectoryDataset(Dataset):
    """Loads images from a flat directory without requiring class subfolders."""
    def __init__(self, directory: str, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(directory) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in {directory}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return a dummy label to match CIFAR10 tuple structure (image, label)
        return image, 0 

class Shape():

    def __init__(self, channels: int, width: int, height: int) -> None:
        """assumes square images for now"""

        self.channels: int = channels
        self.width: int = width
        self.height: int = height
        self.size: int = width
        self.total_dimension: int = channels * width * height

class DataProvider():

    def __init__(self, args: argparse.Namespace) -> None:
        """Currently only provides Cifar10"""

        self.args = args

        if args.dataset == 'cifar10':
            # channels, width, height
            self.data_dims: Shape = Shape(3, 32, 32)

    def get_datasets_for_training(self) -> tuple[DataLoader, DataLoader]:

        transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Scales to [0, 1]
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Shifts to [-1, 1]
        ])

        training_data = datasets.CIFAR10(
            root=self.args.data_dir if self.args.where == 'cluster' else "../data",
            train=True,
            download=True,
            transform=transform,
        )

        test_data = datasets.CIFAR10(
            root=self.args.data_dir if self.args.where == 'cluster' else "../data",
            train=False,
            download=True,
            transform=transform
        )

        # Create data loaders.
        train_dataloader = DataLoader(    # type: ignore
            training_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.args.where == 'cluster' else False
        )
        test_dataloader = DataLoader(    # type: ignore
            test_data,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True if self.args.where == 'cluster' else False
        )

        return train_dataloader, test_dataloader


    def get_dataset_for_eval(self, path_to_generated_samples: str) -> tuple[DataLoader, DataLoader]:

        # InceptionV3 operates optimally on 299x299 images
        transform = Compose([
            Resize((299, 299)),
            ToTensor() 
        ])

        eval_set = datasets.CIFAR10(
            root=self.args.data_dir if self.args.where == 'cluster' else '../data',
            train=False,
            download=True,
            transform=transform
        )

        # Validate requested sample size
        if len(eval_set) < self.args.num_samples:
            raise ValueError(f"Requested {self.args.num_samples} samples, but CIFAR10 test set only has {len(eval_set)}.")
        eval_set = Subset(eval_set, range(self.args.num_samples))    # type: ignore

        dataset_loader = DataLoader(eval_set, batch_size=128, num_workers=4)    # type: ignore


        generated_set_full = FlatDirectoryDataset(directory=path_to_generated_samples, transform=transform)

        if len(generated_set_full) < self.args.num_samples:
            raise ValueError(f"Requested {self.args.num_samples} samples, but fake directory only has {len(generated_set_full)}.")

        # Slice down to exactly `num_samples`
        generated_set = Subset(generated_set_full, range(self.args.num_samples))    # type: ignore

        generated_set_loader = DataLoader(generated_set, batch_size=128, num_workers=4)    # type: ignore


        return dataset_loader, generated_set_loader