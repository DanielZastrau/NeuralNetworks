import argparse

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.dataHandling import DataProvider


def evaluate_fid(args: argparse.Namespace, data: DataProvider, path_to_generated_samples: str, feature_dim: int = 2048):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(device)

    dataset_loader, generated_set_loader = data.get_dataset_for_eval(path_to_generated_samples=path_to_generated_samples)

    print(f"Extracting features from {args.num_samples} real CIFAR-10 images...")
    for images, _ in dataset_loader:
        images = images.to(device)
        fid.update(images, real=True)

    print(f"Extracting features from {args.num_samples} generated images...")
    for images, _ in generated_set_loader:
        images = images.to(device)
        fid.update(images, real=False)

    print("Computing final FID score...")
    fid_score = fid.compute()
    
    return fid_score.item()

def eval_wrapper(args: argparse.Namespace, data: DataProvider, img_path: str):
    
    try:
        score = evaluate_fid(args=args, data=data, path_to_generated_samples=img_path)
        print(f"FID Score ({args.num_samples} samples): {score:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")