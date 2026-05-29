import argparse
import os

import torch
from torchvision.utils import make_grid, save_image    # type: ignore

import matplotlib.pyplot as plt

from utils.diffusion import f, g, b

from utils.sample_kac import TorchKacConstantSampler
from utils.dataHandling import DataProvider

@torch.inference_mode()
def sample(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider) -> torch.Tensor:
    """Euler-Maruyama sampling of the reverse SDE"""

    print(f"Sampling {args.num_samples} images using Euler-Maruyama SDE sampling...")
    device = next(model.parameters()).device

    # 1. Properly scale continuous time from T down to epsilon
    # ! This needs to match the training setup, i.e. if training was on (1e-3, 1) then sampling also needs to be on (1e-3, 1)
    # * 1e-5 as specified by "Song et al 2021 - Score based generative modelling through sdes" and as referenced by "Duong Chemseddine 2025 - Telegraphers Generative Model via Kac Flows"
    epsilon = 1e-5
    time_steps = torch.linspace(1.0, epsilon, args.num_steps, device=device)
    dt = (1.0 - epsilon) / args.num_steps

    # Initialize x with random noise from the prior distribution
    x = torch.randn(args.num_samples, data.data_dims.channels, data.data_dims.width, data.data_dims.height, device=device)

    for step_idx, t_val in enumerate(time_steps):
        if step_idx % 100 == 0 or len(time_steps) - step_idx <= 20:
            print(f"Step {step_idx}/{args.num_steps}")

        # Broadczast the continuous time value to the batch size
        t = torch.ones(args.num_samples, device=device) * t_val

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
        if step_idx == args.num_steps - 1:
            noise_injection = 0.0

        # Continuous SDE reverse step formula
        x = x - drift_update + score_update + noise_injection

    return x

def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider, sampler: TorchKacConstantSampler | None, save_path: str):
    """
    the sampler argument only exists so that the full wrapper does not show a warning for a missing argument
    """

    # generate 64 images
    samples = sample(args=args, model=model, data=data)
    print(f"Generated samples shape: {samples.shape}")  # Should be (64, 3, 32, 32)

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