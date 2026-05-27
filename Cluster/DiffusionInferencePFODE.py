"""Diffusion Inference using Probability Flow ODE as described in Song et al. 2021"""

import argparse

from scipy.integrate import solve_ivp
import numpy as np

import torch

from Diffusion import f, g, b

import matplotlib.pyplot as plt
from torchvision.utils import make_grid # type=ignore

@torch.inference_mode()
def sample(args: argparse.Namespace, model: torch.nn.Module, batch_size: int = 64) -> torch.Tensor:
    print(f"Sampling {batch_size} images using adaptive Probability Flow ODE (RK45)...")
    device = next(model.parameters()).device
    
    # The authors recommend 1e-5 or 1e-3 for adaptive ODE sampling
    epsilon = 1e-5
    shape = (batch_size, 1, 28, 28)
    
    # Initialize x with random noise and flatten for SciPy
    x = torch.randn(shape, device=device)
    x_flat = x.cpu().numpy().flatten()

    def ode_func(t: float, x_flat_numpy: np.ndarray) -> np.ndarray:
        print(f"\rCurrent t: {t:.6f}", end="\r")

        # SciPy provides 't' as a float and 'x' as a 1D numpy array
        x_tensor = torch.from_numpy(x_flat_numpy).float().view(shape).to(device)
        t_tensor = torch.ones(batch_size, device=device) * t
        
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

def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module) -> torch.Tensor:
    
    # generate 64 images
    samples = sample(args=args, model=model, batch_size=64)
    print(f"Generated samples shape: {samples.shape}")  # Should be (64, 1, 28, 28)

    # if your images are normalized to [-1, 1], rescale to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    grid = make_grid(samples, nrow=8, padding=8, normalize=False)

    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")

    path_to_save = f"./{args.where}_{args.which}_{args.epochs}_samples_8x8_PFODE_a{args.abs_tol}_r{args.rel_tol}.png"
    if args.where == 'cluster': path_to_save = f"/homes/math/zastrau/NeuralNetworkSamples/{path_to_save}"
    plt.savefig(path_to_save, dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved generated samples to {path_to_save}")

    plt.close()