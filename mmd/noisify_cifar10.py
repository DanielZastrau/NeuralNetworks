import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import matplotlib.pyplot as plt

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # Scales to [0, 1]
])

training_data = datasets.CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

# Create data loaders.
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)


def noisify(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """This only puts noise on top, but as I also already saw, that leaves fragments of the original image.
    Thus as in kac, mean reverting process"""

    b = 2
    U = torch.rand(x_0.shape)
    width = b * (1 - torch.exp(- t / b))

    # ^ old naive approach
    # return x_0 + U * width

    # ^ mean reverting approach
    return (1 - t) * x_0 + U * width


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

        img = x_plot.detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

fig.suptitle("Noisification of two Cifar10 samples over time using MMD Gradient Flow", fontsize=16)
plt.savefig("mmdAppliedToCifar10.png", dpi=200, bbox_inches="tight")
plt.close(fig)