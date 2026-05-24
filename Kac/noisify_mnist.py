import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2

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

def noisify(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # closed form solution
    return f(t) * x_0 + g(t) * torch.randn_like(x_0)

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