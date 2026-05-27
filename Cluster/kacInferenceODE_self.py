import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid    # type: ignore

from utils.sample_kac import TorchKacConstantSampler

from scipy.integrate import solve_ivp    # type: ignore
import numpy as np

@torch.inference_mode()
def sample(args: argparse.Namespace, model: torch.nn.Module, sampler: TorchKacConstantSampler, batch_size: int = 64) -> torch.Tensor:
    print(f"Sampling {batch_size} images using adaptive Probability Flow ODE (RK45)...")
    device = next(model.parameters()).device
    
    # The authors recommend 1e-5 or 1e-3 for adaptive ODE sampling
    shape = (batch_size, 3, 32, 32)
    
    # Initialize x with random kac noise and flatten for SciPy
    t = torch.ones(batch_size, 1, device=device) * args.T
    x = sampler.sample(t.squeeze(1), dim=3*32*32).to(device)
    x_flat = x.cpu().numpy().flatten()

    def ode_func(t: float, x_flat_numpy: np.ndarray) -> np.ndarray:
        print(f"\rCurrent t: {t:.6f}", end="\r")

        # SciPy provides 't' as a float and 'x' as a 1D numpy array
        x_tensor = torch.from_numpy(x_flat_numpy).float().view(shape).to(device)    # type: ignore
        t_tensor = torch.ones(batch_size, device=device) * t
        
        pred_velocity = model(x_tensor, t_tensor)

        # Calculate ODE derivative: dx/dt = f(x,t) - 0.5 * g(t)^2 * score
        dx_dt = pred_velocity
        
        # Flatten back to numpy for SciPy
        return dx_dt.cpu().numpy().flatten()

    # solve_ivp integrates from t_span[0] to t_span[1]
    # Passing (1.0, epsilon) implicitly handles the negative dt of backward integration
    solution = solve_ivp(
        fun=ode_func,
        t_span=(1.0, 0),
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

def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, sampler: TorchKacConstantSampler):
    """
    The sampler argument only exists so that the full wrapper does not show a warning
    """

    # generate 64 images
    samples = sample(args=args, model=model, batch_size=64, sampler=sampler)
    print(f"Generated samples shape: {samples.shape}")

    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    grid = make_grid(samples, nrow=8, padding=8, normalize=False)

    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")

    path_to_save = f"./{args.where}_{args.which}_{args.epochs}_samples_8x8_fODE_a{args.abs_tol}_r{args.rel_tol}.png"
    if args.where == 'cluster': path_to_save = f"/homes/math/zastrau/NeuralNetworkSamples/{path_to_save}"
    plt.savefig(path_to_save, dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved generated samples to {path_to_save}")

    plt.close()