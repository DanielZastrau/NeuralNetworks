from scipy.integrate import solve_ivp
import numpy as np

import torch

from DiffusionUNetExplainAI import Unet
from Diffusion import beta, f, g, h, b

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

@torch.inference_mode()
def sample_ode_adaptive(batch_size: int = 64) -> torch.Tensor:
    print(f"Sampling {batch_size} images using adaptive Probability Flow ODE (RK45)...")
    device = next(model.parameters()).device
    
    # The authors recommend 1e-5 or 1e-3 for adaptive ODE sampling
    epsilon = 1e-3
    shape = (batch_size, 1, 28, 28)
    
    # Initialize x with random noise and flatten for SciPy
    x = torch.randn(shape, device=device)
    x_flat = x.cpu().numpy().flatten()

    def ode_func(t: float, x_flat_numpy: np.ndarray) -> np.ndarray:
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
        rtol=1e-5,
        atol=1e-5
    )

    print(f"ODE Solver completed in {solution.nfev} neural network evaluations.")
    
    # Extract the final state (last column of solution.y) and reshape
    x_final_flat = solution.y[:, -1]
    x_final = torch.from_numpy(x_final_flat).float().view(shape).to(device)
    
    return x_final

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet().to(device)
    model.load_state_dict(torch.load("/work/zastrau/diffusion/model.pth"))

    N = 2000

    # generate 64 images
    samples = sample_ode_adaptive(batch_size=64)
    print(f"Generated samples shape: {samples.shape}")  # Should be (64, 1, 28, 28)

    # if your images are normalized to [-1, 1], rescale to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    grid = make_grid(samples, nrow=8, padding=2, normalize=False)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.savefig("/work/zastrau/diffusion/diffusion_samples_8x8.png", dpi=200, bbox_inches="tight", pad_inches=0)
    print("Saved generated samples to /work/zastrau/diffusion/diffusion_samples_8x8.png")
    plt.close()