import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from Cluster.utils.dataHandling import DataProvider

class Uint8DatasetWrapper(Dataset):
    """
    Wraps an existing dataset that yields [0, 1] float tensors and converts 
    them to discrete [0, 255] uint8 tensors required by torch_fidelity.
    """
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        # Quantize the continuous [0, 1] float tensor to an 8-bit integer
        img_uint8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8)
        return img_uint8


class Uint8Dataset(Dataset):
    """Wrap a uint8 tensor for FID evaluation."""
    def __init__(self, tensor_uint8: torch.Tensor):
        self.data = tensor_uint8

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx: int):
        return self.data[idx]


def to_uint8_rgb(imgs: torch.Tensor, data: DataProvider) -> torch.Tensor:
    imgs = (imgs + 1) * 0.5
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(data.data_dims.height, data.data_dims.width), mode='bilinear', align_corners=False)
    return (imgs * 255).round().clamp(0, 255).to(torch.uint8)