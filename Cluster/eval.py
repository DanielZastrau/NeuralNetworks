import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchmetrics.image.fid import FrechetInceptionDistance

def evaluate_fid(real_dir: str, fake_dir: str, batch_size: int = 32, feature_dim: int = 2048):
    """
    Computes the FID score between two directories of images.
    
    Args:
        real_dir: Path to the directory containing real images.
        fake_dir: Path to the directory containing generated (fake) images.
        batch_size: Batch size for processing the images.
        feature_dim: InceptionV3 feature layer dimension (64, 192, 768, or 2048).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the metric. 
    # normalize=True allows passing float tensors in the range [0, 1].
    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(device)

    # InceptionV3 traditionally operates on 299x299 images.
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor() 
    ])

    real_dataset = ImageFolder(root=real_dir, transform=transform)
    fake_dataset = ImageFolder(root=fake_dir, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, num_workers=4)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, num_workers=4)

    print("Extracting features from real images...")
    for batch, _ in real_loader:
        images = batch.to(device)
        fid.update(images, real=True)

    print("Extracting features from generated images...")
    for batch, _ in fake_loader:
        images = batch.to(device)
        fid.update(images, real=False)

    print("Computing final FID score...")
    fid_score = fid.compute()
    
    return fid_score.item()

if __name__ == "__main__":
    # Example usage:
    # Set paths to your dataset directories. 
    # ImageFolder expects subdirectories (e.g., path/to/real/class_name/image.png)
    REAL_IMG_PATH = "./data/real_images" 
    FAKE_IMG_PATH = "./data/generated_images"
    
    try:
        score = evaluate_fid(REAL_IMG_PATH, FAKE_IMG_PATH, batch_size=32)
        print(f"FID Score: {score:.4f}")
    except FileNotFoundError as e:
        print(f"Error loading directories: {e}. Ensure paths contain valid image subfolders.")