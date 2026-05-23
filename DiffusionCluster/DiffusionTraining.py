from collections.abc import Callable

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from DiffusionUNetExplainAI import Unet

# positive increasing time schedule
def beta(t: torch.Tensor) -> torch.Tensor:
    """In Sohl-Dickstein 2015 they use a linear schedule with a step of 1 / T"""

    t = torch.as_tensor(t, dtype=torch.float32, device=t.device)
    beta_start = torch.tensor(0.01, dtype=t.dtype, device=t.device)
    beta_end = torch.tensor(20.0, dtype=t.dtype, device=t.device)

    # linear interpolation between beta_start and beta_end
    return beta_start + t * (beta_end - beta_start)

    # # Sohl-Dickstein 2015's linear schedule with a step of 1 / T
    # return (beta_start + (t / T) * (beta_end - beta_start))

def f(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    return -0.5 * beta(t) * x_t

def g(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(beta(t))

def h(t: torch.Tensor) -> torch.Tensor:
    """integral from 0 to t of beta(s) ds, which is used in the closed-form solution of the SDE
    Using Sohl-Dickstein 2015's linear schedule, we can compute this integral in closed form as:
    h(t) = (beta_start * t + 0.5 * (beta_end - beta_start) * t^2) / T
    """
    beta_start = torch.tensor(0.01, dtype=t.dtype, device=t.device)
    beta_end = torch.tensor(20.0, dtype=t.dtype, device=t.device)

    # Sohl-Dickstein 2015's linear schedule with a step of 1 / T
    return torch.as_tensor(t*(beta_start + (t / 2) * (beta_end - beta_start)), dtype=torch.float32, device=t.device)

def b(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-h(t) / 2)

def noisify(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # closed form solution
    return torch.sqrt(1 - b(t)**2) * torch.randn_like(x_0) + b(t) * x_0

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
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):
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
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {batch}/{len(dataloader)}, LR: {current_lr:.6f}, Loss: {loss.item():.6f}", end="\r")

    print(f"\n Train Avg loss: {train_loss / len(dataloader):>8f} \n")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model and move to device
    model = Unet().to(device)

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), # Scales to [0, 1]
        v2.Grayscale(num_output_channels=1),
        v2.Normalize(mean=[0.5], std=[0.5])    # Shifts to [-1, 1]
    ])

    training_data = datasets.MNIST(
        root="/fast/zastrau/data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="/fast/zastrau/data",
        train=False,
        download=True,
        transform=transform
    )

    batch_size = 512

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    # Define step counts
    epochs = 40
    batches_per_epoch = len(train_dataloader)
    total_steps = epochs * batches_per_epoch

    # Dedicate the first 5% of total iterations to warming up
    warmup_steps = int(total_steps * 0.05) 

    # 1. Initialize optimizer with the TARGET scaled learning rate
    target_lr = 5.6e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr, weight_decay=1e-4)

    # 2. Warmup: Linearly increase LR from near-zero (target_lr * 1e-8) to target_lr
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-8, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )

    # 3. Decay: Cosine annealing for the remainder of training
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=(total_steps - warmup_steps), 
        eta_min=1e-5
    )

    # 4. Chain the schedulers
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    
    # train the model
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train(train_dataloader, model, loss_fn, optimizer, scheduler)
        test(test_dataloader, model, loss_fn)

        print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")

        torch.save(model.state_dict(), "/work/zastrau/diffusion/model.pth")
    print("Done!")