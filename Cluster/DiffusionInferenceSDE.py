import argparse

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from Diffusion import f, g, b

@torch.inference_mode()
def sample(model: torch.nn.Module, batch_size: int = 64, num_steps: int = 1000) -> torch.Tensor:
    """Euler-Maruyama sampling of the reverse SDE"""

    print(f"Sampling {batch_size} images using Euler-Maruyama SDE sampling...")
    device = next(model.parameters()).device

    # 1. Properly scale continuous time from T down to epsilon (e.g., 1.0 down to 1e-5 or 1e-3)
    # Match the bounds used during your training phase!
    epsilon = 1e-5
    time_steps = torch.linspace(1.0, epsilon, num_steps, device=device)
    dt = (1.0 - epsilon) / num_steps

    # Initialize x with random noise from the prior distribution
    x = torch.randn(batch_size, 3, 32, 32, device=device)

    for step_idx, t_val in enumerate(time_steps):
        if step_idx % 100 == 0 or len(time_steps) - step_idx <= 20:
            print(f"Step {step_idx}/{num_steps}")

        # Broadczast the continuous time value to the batch size
        t = torch.ones(batch_size, device=device) * t_val

        # Get continuous coefficients
        f_t_x = f(t, x)
        g_t = g(t).view(-1, 1, 1, 1)
        b_t = b(t).view(-1, 1, 1, 1)

        # Predict score using continuous time
        pred_noise = model(x, t)
        pred_score = - pred_noise / torch.sqrt(1 - b_t**2)

        # 2. Scale updates explicitly by dt and sqrt(dt)
        drift_update = f_t_x * dt
        score_update = (g_t ** 2) * pred_score * dt
        noise_injection = g_t * torch.sqrt(torch.tensor(dt, device=device)) * torch.randn_like(x)

        # Don't add random noise at the very last step
        if step_idx == num_steps - 1:
            noise_injection = 0.0

        # Continuous SDE reverse step formula
        x = x - drift_update + score_update + noise_injection

    return x

def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module) -> torch.Tensor:

    # generate 64 images
    samples = sample(model=model, batch_size=64, num_steps=args.num_steps)
    print(f"Generated samples shape: {samples.shape}")  # Should be (64, 1, 28, 28)

    # if your images are normalized to [-1, 1], rescale to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    grid = make_grid(samples, nrow=8, padding=2, normalize=False)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")

    path_to_save = f"./{args.where}_{args.which}_{args.epochs}_samples_8x8_SDE.png"
    if args.where == 'cluster': path_to_save = f"/homes/math/zastrau/NeuralNetworkSamples/{path_to_save}"
    plt.savefig(path_to_save, dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved generated samples to {path_to_save}")
    
    plt.close()