"""Instead of loading data in the seperate files, load it centrally here, so that data handling only has to be managed in one location"""

import argparse

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

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

        if not 'horizontal_flips' in self.args:
            self.horizontal_flipping = False
        else:
            self.horizontal_flipping = args.horizontal_flips

        # channels, width, height
        self.data_dims: Shape = Shape(3, 32, 32)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def transform(self):

        if self.args.horizontal_flips:
            return v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=self.args.horizontal_flips_p),
                v2.ToDtype(torch.float32, scale=True), # scales to [0,1]
                v2.Normalize((0.5,) * 3, (0.5,) * 3), # scales to [-1, 1]
            ])
        else:
            return v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True), # scales to [0,1]
                v2.Normalize((0.5,) * 3, (0.5,) * 3), # scales to [-1, 1]
            ])

    def get_datasets_for_training(self) -> tuple[DataLoader, DataLoader]:

        batch_size = self.args.training_batch_size

        training_data = datasets.CIFAR10(root='/work/zastrau/cifar10', train=True, download=True, transform=self.transform())
        test_data = datasets.CIFAR10(root='/work/zastrau/cifar10', train=False, download=True, transform=self.transform())

        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=1, drop_last=True)

        return train_dataloader, test_dataloader


    def get_dataset_for_full_eval(self) -> DataLoader:
        from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb

        eval_set = datasets.CIFAR10(root='/work/zastrau/cifar10', train=True, download=True, transform=self.transform())

        # Validate requested sample size
        if len(eval_set) < self.args.eval_num_samples:
            raise ValueError(f"Requested {self.args.eval_num_samples} samples, but CIFAR10 test set only has {len(eval_set)}.")
        eval_set = Subset(eval_set, range(self.args.eval_num_samples))

        dataset_loader = DataLoader(eval_set, batch_size=512, shuffle=False, num_workers=4)

        real_images = []
        for (imgs, _) in dataset_loader:
            real_images.append(to_uint8_rgb(imgs.to(self.device).cpu(), self))
            if sum(x.size(0) for x in real_images) >= self.args.eval_num_samples:
                break
        real_images = torch.cat(real_images)[:self.args.eval_num_samples].cpu()
        real_ds = Uint8Dataset(real_images)

        return real_ds

    
    def get_dataset_for_periodic_eval(self) -> DataLoader:
        from Cluster.utils.uint8_utils import to_uint8_rgb, Uint8Dataset

        batch_size = self.args.training_batch_size

        real_ds = datasets.CIFAR10(root='/work/zastrau/cifar10', train=True, download=True, transform=self.transform())

        real_ds_loader = DataLoader(real_ds, batch_size=batch_size * 4, shuffle=False, num_workers=1)

        real_images = []
        for (imgs, _) in real_ds_loader:
            real_images.append(to_uint8_rgb(imgs.to(self.device), self))
            if sum(x.size(0) for x in real_images) >= self.args.training_evaluation_period_fid_num_samples:
                break
        real_images = torch.cat(real_images)[:self.args.training_evaluation_period_fid_num_samples].cpu()
        real_ds = Uint8Dataset(real_images)

        return real_ds