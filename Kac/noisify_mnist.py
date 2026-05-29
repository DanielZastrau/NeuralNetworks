import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import math

import matplotlib.pyplot as plt

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

# Create data loaders.
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)


def f(t: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(t) - t

def g(t: torch.Tensor) -> torch.Tensor:
    return t**2


def kac_path(t: torch.Tensor, wave_front_speed: float = 3.0, change_rate: float = 2.0, shape: tuple = None, device: torch.device = None) -> torch.Tensor:
    """
    Vectorized Mean-Reverting Kac Process for arbitrary tensor shapes.
    """
    if shape is None:
        raise ValueError("Target shape must be provided.")
        
    # 1. Handle t broadcasting and device mapping
    if isinstance(t, torch.Tensor):
        device = t.device if device is None else device
        t_max = t.max().item()
        # Broadcast 1D t (e.g., batch_size) to target shape (e.g., [batch_size, C, H, W])
        if t.ndim == 1 and len(shape) > 1:
            view_shape = [-1] + [1] * (len(shape) - 1)
            t_broadcast = t.view(*view_shape)
        else:
            t_broadcast = t
    else:
        device = torch.device('cpu') if device is None else device
        t_max = float(t)
        t_broadcast = torch.tensor(t, device=device)

    # 2. Determine max jumps needed (Mean + 5 Standard Deviations buffer)
    max_events = int((change_rate * t_max) + 5 * math.sqrt(change_rate * t_max) + 10)

    # 3. Generate inter-arrival times for all dimensions simultaneously
    # Shape: [max_events, *shape]
    inter_arrivals = torch.empty((max_events, *shape), device=device).exponential_(1.0 / change_rate)

    # 4. Convert to absolute arrival times
    arrival_times = torch.cumsum(inter_arrivals, dim=0)

    # 5. Clamp times to maximum t. 
    # Any event occurring after t is clamped to t, making its subsequent duration 0.
    clamped_times = torch.clamp(arrival_times, max=t_broadcast)

    # 6. Prepend t=0 to calculate durations
    zeros = torch.zeros((1, *shape), device=device)
    full_times = torch.cat([zeros, clamped_times], dim=0)

    # 7. Calculate time spent in each state (dt)
    durations = full_times[1:] - full_times[:-1]

    # 8. Create alternating signs for the integrand (+1, -1, +1, -1...)
    signs = torch.ones((max_events, *shape), device=device)
    signs[1::2] = -1.0

    # 9. Sample initial random directions D_0 in {-1, 1}
    initial_directions = torch.randint(0, 2, shape, device=device).float() * 2.0 - 1.0

    # 10. Integrate: sum of (duration * sign) * velocity * initial_direction
    integral = torch.sum(durations * signs, dim=0)
    kac_values = initial_directions * wave_front_speed * integral

    return kac_values


def noisify(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Mean Reverting forward noise process"""
    
    # Kac process now natively handles the shape and outputs a matching tensor
    noise = kac_path(t=g(t), shape=x0.shape, device=x0.device)
    
    # Broadcast f(t) if t is a 1D batch tensor
    if isinstance(t, torch.Tensor) and t.ndim == 1:
        f_t = f(t).view(-1, *([1] * (x0.ndim - 1)))
    else:
        f_t = f(t)

    return f_t * x0 + noise

# plot noisified samples at different time steps (simple 2-row visualization)
times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
sample_indices = [0, 1]
samples = [training_data[i][0] for i in sample_indices]
labels = [training_data[i][1] for i in sample_indices]

# Create 2 rows x len(times) columns
n_cols = len(times)
fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(n_cols * 3, 6))

for row in range(2):
    x_0 = samples[row]
    for col, t_val in enumerate(times):
        ax = axes[row, col]
        t_item = t_val.item()
        if t_item == 0.0:
            x_plot = x_0
            title = f"t=0.00\nlabel={labels[row]}"
        elif t_item == 1.0:
            x_plot = noisify(x_0, t_val)
            title = f"t={t_item:.2f}\nnoise"
        else:
            x_plot = noisify(x_0, t_val)
            title = f"t={t_item:.2f}"

        img = x_plot.squeeze().detach().cpu().numpy()
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")

fig.suptitle("Noisification of two MNIST samples over time", fontsize=16)
plt.savefig("kac_applied_to_mnist.png", dpi=200, bbox_inches="tight")
plt.close(fig)