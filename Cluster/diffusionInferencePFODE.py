"""Diffusion Inference using Probability Flow ODE as described in Song et al. 2021"""

import argparse
import os

import torch
import numpy as np
from scipy.integrate import solve_ivp    # type: ignore
from torchvision.utils import make_grid, save_image    # type: ignore
import matplotlib.pyplot as plt

from Cluster.utils.diffusion import f, g, b


from Cluster.utils.sample_kac import TorchKacConstantSampler    # only imported for uniform typing
from Cluster.utils.dataHandling import DataProvider

@torch.inference_mode()
def sample(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider) -> torch.Tensor:
    print(f"Sampling {args.num_samples} images using adaptive Probability Flow ODE (RK45)...")
    device = next(model.parameters()).device

    # The authors recommend 1e-5 or 1e-3 for adaptive ODE sampling
    epsilon = 1e-5
    shape = (args.num_samples, data.data_dims.channels, data.data_dims.width, data.data_dims.height)
    
    # Initialize x with random noise and flatten for SciPy
    x = torch.randn(shape, device=device)
    x_flat = x.cpu().numpy().flatten()

    def ode_func(t: float, x_flat_numpy: np.ndarray) -> np.ndarray:
        print(f"\rCurrent t: {t:.6f}", end="\r")

        # SciPy provides 't' as a float and 'x' as a 1D numpy array
        x_tensor = torch.from_numpy(x_flat_numpy).float().view(shape).to(device)
        t_tensor = torch.ones(args.num_samples, device=device) * t
        
        # Get continuous coefficients
        f_t_x = f(t_tensor, x_tensor)
        g_t = g(t_tensor).view(-1, 1, 1, 1)
        b_t = b(t_tensor).view(-1, 1, 1, 1)
        
        # Predict score
        pred_noise = model(x_tensor, t_tensor)
        pred_score = -pred_noise / torch.sqrt(1 - b_t**2)
        
        # Calculate ODE derivative: dx/dt = f(x,t) - 0.5 * g(t)^2 * score
        dx_dt = f_t_x - 0.5 * (g_t ** 2) * pred_score
        
        # Flatten back to numpy for SciPy
        return dx_dt.cpu().numpy().flatten()

    # solve_ivp integrates from t_span[0] to t_span[1]
    # Passing (1.0, epsilon) implicitly handles the negative dt of backward integration
    solution = solve_ivp(
        fun=ode_func,
        t_span=(1.0, epsilon),
        y0=x_flat,
        method='RK45',
        rtol=args.rel_tol,
        atol=args.abs_tol
    )

    print(f"ODE Solver completed in {solution.nfev} neural network evaluations.")
    
    # Extract the final state (last column of solution.y) and reshape
    x_final_flat = solution.y[:, -1]
    x_final = torch.from_numpy(x_final_flat).float().view(shape).to(device)
    
    return x_final

def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider, sampler: TorchKacConstantSampler | None, save_path: str):
    """
    The sampler argument only exists so that the full wrapper does not show a warning
    """

    # generate 64 images
    samples = sample(args=args, model=model, data=data)
    print(f"Generated samples shape: {samples.shape}")

    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)


    if args.sampler_mode == '8x8':
        grid = make_grid(samples, nrow=8, padding=2, normalize=False)

        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
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