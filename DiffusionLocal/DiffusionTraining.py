from collections.abc import Callable

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F

from DiffusionLocal.DiffusionNeuralNetworkLarge import Unet
from Diffusion import b

def loss_fn(model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:

    x_0 = mini_batch

    # sample time steps t uniformly from [0, 1)
    t = torch.rand(mini_batch.size(0), device=mini_batch.device)

    # draw standard gaussian noise matching the shape of x_0
    noise = torch.randn_like(x_0)

    # compute b(t)
    b_t = b(t)
    b_t = b_t.view(-1, 1, 1, 1)  # Reshape b_t to (batch_size, 1, 1, 1) for broadcasting

    # compute x
    x = b_t * x_0 + torch.sqrt(1 - b_t**2) * noise

    # simplyfiy the target
    # target = - noise / torch.sqrt(1 - b_t**2)

    # simplyfiy the target even further and only learn the noise and scale to the score later during inference
    target = noise

    # evaluate the neural network score prediction
    pred = model(x, t)

    # compute empirical loss
    loss = F.mse_loss(pred, target)

    return loss

def train(dataloader, model: torch.nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          optimizer: torch.optim.Optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    train_loss = 0

    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)

        loss = loss_fn(model=model, mini_batch=X)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 100 == 0:
            print(f"batch: {batch}, loss: {loss.item():>7f}", end="\r")

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


if __name__ == "__main__":
    # model = ConditionalUNet(n_channels=1, n_classes=1)
    model = Unet()

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
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # train the model
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        scheduler.step()
        print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")
        torch.save(model.state_dict(), "model.pth")
    print("Done!")
