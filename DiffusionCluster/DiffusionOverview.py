import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import matplotlib.pyplot as plt

from Diffusion.Diffusion import beta, f, g, h, b

transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True), # Scales to [0, 1]
    v2.Grayscale(num_output_channels=1),
    v2.Normalize(mean=[0.5], std=[0.5])    # Shifts to [-1, 1]
])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Create data loaders.
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def noisify(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # closed form solution
    return torch.sqrt(1 - b(t)**2) * torch.randn_like(x_0) + b(t) * x_0

# plot beta, f, g, h, b, and noisified samples at different time steps
t = torch.linspace(0.0, 1.0, 200)
beta_t = beta(t)
x_t = torch.tensor(1.0, dtype=torch.float32)
f_t = f(t, x_t)
g_t = g(t)
h_t = h(t)
b_t = b(t)

times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
sample_indices = [0, 1]
samples = [training_data[i][0] for i in sample_indices]
labels = [training_data[i][1] for i in sample_indices]

fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(nrows=5, ncols=5, hspace=0.35, wspace=0.2)

ax_beta = fig.add_subplot(gs[0, 0])
ax_f = fig.add_subplot(gs[0, 1])
ax_g = fig.add_subplot(gs[1, 0])
ax_h = fig.add_subplot(gs[1, 1])
ax_b = fig.add_subplot(gs[2, 0])
ax_empty = fig.add_subplot(gs[2, 1])
ax_empty.axis("off")
ax_beta.plot(t.numpy(), beta_t.numpy(), color="blue")
ax_beta.set_title(r"$\beta(t)$")
ax_beta.set_xlabel("t")
ax_beta.set_ylabel(r"$\beta(t)$")
ax_beta.grid(True)

ax_f.plot(t.numpy(), f_t.numpy(), color="green")
ax_f.set_title(r"$f(t, x)$ with $x = 1.0$")
ax_f.set_xlabel("t")
ax_f.set_ylabel(r"$f(t, x)$")
ax_f.grid(True)

ax_g.plot(t.numpy(), g_t.numpy(), color="red")
ax_g.set_title(r"$g(t)$")
ax_g.set_xlabel("t")
ax_g.set_ylabel(r"$g(t)$")
ax_g.grid(True)

ax_h.plot(t.numpy(), h_t.numpy(), color="purple")
ax_h.set_title(r"$h(t)$")
ax_h.set_xlabel("t")
ax_h.set_ylabel(r"$h(t)$")
ax_h.grid(True)

ax_b.plot(t.numpy(), b_t.numpy(), color="orange")
ax_b.set_title(r"$b(t)$")
ax_b.set_xlabel("t")
ax_b.set_ylabel(r"$b(t)$")
ax_b.grid(True)

for row in range(2):
    x_0 = samples[row]
    for col, t_val in enumerate(times):
        ax = fig.add_subplot(gs[3 + row, col])
        if t_val == 0.0:
            x_plot = x_0
            title = f"t=0.00\nlabel={labels[row]}"
        elif t_val == 1.0:
            x_plot = noisify(x_0, t_val)
            title = f"t={t_val:.2f}\nnoise"
        else:
            x_plot = noisify(x_0, t_val)
            title = f"t={t_val:.2f}"
        img = x_plot.squeeze().detach().cpu().numpy()
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")

fig.suptitle("Diffusion Overview: schedules and sample transition to noise", fontsize=20)
plt.savefig("diffusion_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)