import torch

from DiffusionModel import ConditionalUNet


model = ConditionalUNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load("model.pth"))

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
    t = t.view(-1, 1, 1, 1)  # Reshape t to (B, 1, 1, 1) for broadcasting
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


@torch.inference_mode()
def sample(batch_size: int = 64, T: int = 1000) -> torch.Tensor:       
    print(f"Sampling {batch_size} images from the diffusion model...")

    # Initialize x with random noise in image shape (B, C, H, W)
    x = torch.randn(batch_size, 1, 28, 28, device=next(model.parameters()).device)

    # 1. Define step size dt and numerical stability offset eps
    dt = torch.tensor(1.0) / T
    eps = torch.tensor(1e-5)

    # 2. Iterate normalized time backwards from 1.0 down to eps
    time_steps = torch.linspace(1.0, eps, T)

    for i, t_val in enumerate(time_steps):
        if i % 10 == 0:
            print(f"Sampling time step: {i}", end="\r")

        t = torch.ones(batch_size, device=x.device) * t_val  

        f_t_x = f(t, x)
        
        g_t = g(t)
        g_t = g_t.view(-1, 1, 1, 1)  # Reshape g_t to (B, 1, 1, 1) for broadcasting

        pred_score = model(x, t)

        # 3. Apply the appropriate dt and sqrt(dt) scalings to the Euler-Maruyama update
        drift = f_t_x - (g_t**2) * pred_score
        diffusion = g_t
        
        # Notice that we subtract the drift * dt because time is moving backwards
        x = x - drift * dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)
        
    return x

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# generate 64 images
samples = sample(batch_size=64)
print(f"Generated samples shape: {samples.shape}")  # Should be (64, 1, 28, 28)

# if your images are normalized to [-1, 1], rescale to [0, 1]
samples = (samples + 1.0) / 2.0
samples = samples.clamp(0.0, 1.0)

grid = make_grid(samples, nrow=8, padding=2, normalize=False)

plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
plt.axis("off")
plt.savefig("./diffusion_samples_8x8.png", dpi=200, bbox_inches="tight", pad_inches=0)
print("Saved generated samples to ./diffusion_samples_8x8.png")
plt.close()