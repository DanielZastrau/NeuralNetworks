import os
import argparse

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms    # type: ignore
from torchvision.datasets import CIFAR10    # type: ignore
from torchmetrics.image.fid import FrechetInceptionDistance

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

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return a dummy label to match CIFAR10 tuple structure (image, label)
        return image, 0 

def evaluate_fid(fake_dir: str, num_samples: int, batch_size: int = 32, feature_dim: int = 2048):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(device)

    # InceptionV3 operates optimally on 299x299 images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor() 
    ])

    # Load Full Datasets
    print("Loading datasets...")
    real_dataset_full = CIFAR10(root='../data', train=False, download=True, transform=transform)
    fake_dataset_full = FlatDirectoryDataset(directory=fake_dir, transform=transform)

    # Validate requested sample size
    if len(real_dataset_full) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but CIFAR10 test set only has {len(real_dataset_full)}.")
    if len(fake_dataset_full) < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but fake directory only has {len(fake_dataset_full)}.")

    # Slice down to exactly `num_samples`
    real_dataset = Subset(real_dataset_full, range(num_samples))    # type: ignore
    fake_dataset = Subset(fake_dataset_full, range(num_samples))    # type: ignore

    real_loader = DataLoader(real_dataset, batch_size=batch_size, num_workers=4)    # type: ignore
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, num_workers=4)    # type: ignore

    print(f"Extracting features from {num_samples} real CIFAR-10 images...")
    for images, _ in real_loader:
        images = images.to(device)
        fid.update(images, real=True)

    print(f"Extracting features from {num_samples} generated images...")
    for images, _ in fake_loader:
        images = images.to(device)
        fid.update(images, real=False)

    print("Computing final FID score...")
    fid_score = fid.compute()
    
    return fid_score.item()

def eval_wrapper(args: argparse.Namespace, img_path: str):
    
    try:
        score = evaluate_fid(fake_dir=img_path, num_samples=args.num_samples, batch_size=32)
        print(f"FID Score ({args.num_samples} samples): {score:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")