import argparse
import os

import torch
from torchvision.utils import make_grid, save_image    # type: ignore

import matplotlib.pyplot as plt

from utils.diffusion import f, g, b

from utils.sample_kac import TorchKacConstantSampler

@torch.inference_mode()
def sample(model: torch.nn.Module, num_samples: int = 64, num_steps: int = 1000) -> torch.Tensor:
    """Euler-Maruyama sampling of the reverse SDE"""

    print(f"Sampling {num_samples} images using Euler-Maruyama SDE sampling...")
    device = next(model.parameters()).device

    # 1. Properly scale continuous time from T down to epsilon
    epsilon = 1e-5    # 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
    time_steps = torch.linspace(1.0, epsilon, num_steps, device=device)
    dt = (1.0 - epsilon) / num_steps

    # Initialize x with random noise from the prior distribution
    x = torch.randn(num_samples, 3, 32, 32, device=device)

    for step_idx, t_val in enumerate(time_steps):
        if step_idx % 100 == 0 or len(time_steps) - step_idx <= 20:
            print(f"Step {step_idx}/{num_steps}")

        # Broadczast the continuous time value to the batch size
        t = torch.ones(num_samples, device=device) * t_val

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

def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, sampler: TorchKacConstantSampler | None, save_path: str):
    """
    the sampler argument only exists so that the full wrapper does not show a warning for a missing argument
    """

    # generate 64 images
    samples = sample(model=model, num_samples=args.num_samples, num_steps=args.num_steps)
    print(f"Generated samples shape: {samples.shape}")  # Should be (64, 1, 28, 28)

    # if your images are normalized to [-1, 1], rescale to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    if args.sampler_mode == '8x8':
        grid = make_grid(samples, nrow=8, padding=2, normalize=False)

        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
        print(f"Saved generated samples to {save_path}")
        
        plt.close()
    else:    # args.sampler_diff_mode == 'set'
        os.makedirs(save_path, exist_ok=True)
        
        # save_image automatically handles the C, H, W shape and normalizes to 0-255 internally
        for i, img in enumerate(samples):
            img_path = os.path.join(save_path, f"sample_{i:05d}.png")
            save_image(img, img_path)
            
        print(f"Saved {len(samples)} individual images to {save_path}/")