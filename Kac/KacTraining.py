from collections.abc import Callable
import argparse
import time

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F

from KacNeuralNetworkSmall import ConditionalUNet
from utils.sample_kac import TorchKacConstantSampler
from utils.velo_utils import compute_velocity

from Kac import get_f, get_df, get_g, get_dg

_IMG_SIZE = 28
dim = _IMG_SIZE * _IMG_SIZE

def get_args():
    parser = argparse.ArgumentParser(description="Train Kac on MNIST")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--a', type=float, default=9.0)
    parser.add_argument('--c', type=float, default=3.0)
    parser.add_argument('--T', type=float, default=1.0)
    return parser.parse_args()


# def get_unet(img_size: int, channels: int):
#     return UNetModel(
#         in_channels=channels,
#         out_channels=channels,
#         num_res_blocks=2,
#         image_size=img_size,
#         model_channels=128,
#         channel_mult=(1, 2, 2, 2),
#         num_heads=4,
#         num_head_channels=64,
#         dropout=0.1,
#         attention_resolutions=(16,)
#     )

def loss_fn(model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    B = mini_batch.size(0)
    x0 = mini_batch.view(B, -1).to(device)

    # Sample original time t
    t = torch.rand(B, 1, device=device) * args.T
    
    # --- Use g(t) for noise process time ---
    t_for_noise_proc = get_g(t.squeeze(1), args.T)
    
    # Sample noise using g(t)
    tau = sampler.sample(t_for_noise_proc, dim=dim).to(device)
    
    # Use original t for signal schedule f(t)
    f = get_f(t.squeeze(1), args.T).unsqueeze(1)
    xt = f * x0 + tau
    drift = get_df(t.squeeze(1), args.T).unsqueeze(1) * x0

    with torch.no_grad():
        # Compute velocity using g(t)
        velo = get_dg(t, args.T) * compute_velocity((xt - f * x0), t_for_noise_proc.unsqueeze(1), args.a, args.c, epsilon=1e-4)

    # Model is conditioned on original time t
    pred = model(xt.view(B, 1, _IMG_SIZE, _IMG_SIZE), t.squeeze(1)).view(B, -1)
    loss = F.mse_loss(pred, velo + drift)

    return loss

def train(dataloader, model: torch.nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          optimizer: torch.optim.Optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    train_loss = 0

    for batch, (X, _) in enumerate(dataloader):
        t_start = time.time()

        X = X.to(device)
        
        loss = loss_fn(model=model, mini_batch=X)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        t_end = time.time()

        if batch % 100 == 0:
            print(f"batch: {batch}, loss: {loss.item():>7f}, time: {t_end - t_start:.4f}s", end="\r")
    print(f"Train Avg loss: {train_loss / len(dataloader):>8f} \n")

def test(dataloader, model: torch.nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)

            test_loss += loss_fn(model=model, mini_batch=X).item()
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalUNet(n_channels=1, n_classes=1).to(device)

    torch.set_num_threads(5)

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), # Scales to [0, 1]
        v2.Grayscale(num_output_channels=1),
        v2.Normalize(mean=[0.5], std=[0.5])    # Shifts to [-1, 1]
    ])

    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # train the model
    args = get_args()
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

    sampler = TorchKacConstantSampler(a=args.a, c=args.c, T=args.T, M=50000, K=4096)

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        
        scheduler.step()
        print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")
        
        torch.save(model.state_dict(), "model.pth")
    print("Done!")