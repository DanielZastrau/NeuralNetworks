from collections.abc import Callable

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F

from DiffusionModel import ConditionalUNet


model = ConditionalUNet(n_channels=1, n_classes=1)

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

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# cutoff time for the diffusion process
T = 1000

# positive increasing time schedule
def beta(t: torch.Tensor) -> torch.Tensor:
    """In Sohl-Dickstein 2015 they use a linear schedule with a step of 1 / T"""

    global T

    t = torch.as_tensor(t, dtype=torch.float32, device=t.device)
    beta_start = torch.tensor(0.001, dtype=t.dtype, device=t.device)
    beta_end = torch.tensor(0.02, dtype=t.dtype, device=t.device)

    # linear interpolation between beta_start and beta_end
    # return beta_start + t * (beta_end - beta_start)

    # Sohl-Dickstein 2015's linear schedule with a step of 1 / T
    return (beta_start + (t / T) * (beta_end - beta_start))

def f(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    return -0.5 * beta(t) * x_t

def g(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(beta(t))

def h(t: torch.Tensor) -> torch.Tensor:
    """integral from 0 to t of beta(s) ds, which is used in the closed-form solution of the SDE
    Using Sohl-Dickstein 2015's linear schedule, we can compute this integral in closed form as:
    h(t) = (beta_start * t + 0.5 * (beta_end - beta_start) * t^2) / T
    """
    beta_start = torch.tensor(0.001, dtype=t.dtype, device=t.device)
    beta_end = torch.tensor(0.02, dtype=t.dtype, device=t.device)

    return torch.as_tensor(t*(beta_start + (t / 2) * (beta_end - beta_start)), dtype=torch.float32, device=t.device)

def b(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-h(t) / 2)

def noisify(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # closed form solution
    return torch.sqrt(1 - b(t)**2) * torch.randn_like(x_0) + b(t) * x_0

def loss_fn(model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:
    global T

    x_0 = mini_batch

    # sample time steps t uniformly from [0, T]
    t = torch.rand(mini_batch.size(0), device=mini_batch.device) * T
    print(f"t.shape: {t.shape}, x_0.shape: {x_0.shape}")

    # draw standard gaussian noise matching the shape of x_0
    noise = torch.randn_like(x_0)

    # compute b(t)
    b_t = b(t)
    b_t = b_t.view(-1, 1, 1, 1)  # Reshape b_t to (batch_size, 1, 1, 1) for broadcasting

    # compute x
    x = b_t * x_0 + torch.sqrt(1 - b_t**2) * noise

    # simplyfiy the target
    target = noise / torch.sqrt(1 - b_t**2)

    # evaluate the neural network score prediction
    pred = model(x, t)

    # compute empirical loss
    loss = F.mse_loss(pred, target)

    return loss

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Scheduler to reduce the learning rate every 2 epochs by 0.5
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

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
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"batch: {batch}, loss: {loss:>7f}", end="\r")

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


# train the model
torch.autograd.set_detect_anomaly(True)

epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    scheduler.step()
    print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")
    torch.save(model.state_dict(), "model.pth")
print("Done!")


# # ============================================================================

# import matplotlib.pyplot as plt

# # plot beta, f, g
# t = torch.linspace(0.0, T, 200)
# beta_t = beta(t)
# x_t = torch.tensor(1.0, dtype=torch.float32)
# f_t = f(t, x_t)
# g_t = g(t)

# times = torch.tensor([0.0, T/4, T/2, 3*T/4, T])
# sample_indices = [0, 1]
# samples = [training_data[i][0] for i in sample_indices]
# labels = [training_data[i][1] for i in sample_indices]

# fig = plt.figure(figsize=(20, 16))
# gs = fig.add_gridspec(nrows=5, ncols=5, hspace=0.35, wspace=0.2)

# ax_beta = fig.add_subplot(gs[0, :])
# ax_f = fig.add_subplot(gs[1, :])
# ax_g = fig.add_subplot(gs[2, :])

# ax_beta.plot(t.numpy(), beta_t.numpy(), color="blue")
# ax_beta.set_title(r"$\beta(t)$")
# ax_beta.set_xlabel("t")
# ax_beta.set_ylabel(r"$\beta(t)$")
# ax_beta.grid(True)

# ax_f.plot(t.numpy(), f_t.numpy(), color="green")
# ax_f.set_title(r"$f(t, x)$ with $x = 1.0$")
# ax_f.set_xlabel("t")
# ax_f.set_ylabel(r"$f(t, x)$")
# ax_f.grid(True)

# ax_g.plot(t.numpy(), g_t.numpy(), color="red")
# ax_g.set_title(r"$g(t)$")
# ax_g.set_xlabel("t")
# ax_g.set_ylabel(r"$g(t)$")
# ax_g.grid(True)

# for row in range(2):
#     x_0 = samples[row]
#     for col, t_val in enumerate(times):
#         ax = fig.add_subplot(gs[3 + row, col])
#         if t_val == 0.0:
#             x_plot = x_0
#             title = f"t=0.00\nlabel={labels[row]}"
#         else:
#             x_plot = noisify(x_0, t_val)
#             title = f"t={t_val:.2f}"
#         img = x_plot.squeeze().detach().cpu().numpy()
#         ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
#         ax.set_title(title)
#         ax.axis("off")

# fig.suptitle("Diffusion Summary: beta(t), f(t,x), g(t), and MNIST Diffusion Samples", fontsize=18)
# plt.savefig("diffusion_summary.png", dpi=200, bbox_inches="tight")
# plt.close(fig)