import argparse
import os

import torch
from torchdiffeq import odeint    # type: ignore
from torchvision.utils import make_grid, save_image    # type: ignore
from tqdm import tqdm
import matplotlib.pyplot as plt

from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.dataHandling import DataProvider


class ODEWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, data: DataProvider):
        super().__init__()
        self.model = model
        self.nfe = 0
        self.pbar = None

        self.data = data

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        self.nfe += 1
        if self.pbar is not None:
            self.pbar.set_postfix({"t": f"{t.item():.4f}", "NFE": self.nfe})

        B = x.shape[0]
        x_img = x.view(B, self.data.data_dims.channels, self.data.data_dims.width, self.data.data_dims.height)
        t_vec = torch.full((B,), float(t), device=x.device)
        
        v = self.model(x_img, t_vec)
            
        return v.view(x.shape)

def sample_ode(model: torch.nn.Module, x_T: torch.Tensor, T: float, num_steps: int,
               device: torch.device, data: DataProvider, method: str = 'euler', max_batch: int = 512,
               a_tol: float = 1e-4, r_tol: float = 1e-4):
    """
    args:
        model: torch.nn.module - the neural network
        x_T: torch.Tensor - tensor of samples of the noise distribution
    """
    # initialize the ode fn
    ode_fn = ODEWrapper(model, data).to(device)

    # track solution paths for later use
    traj: list[torch.Tensor] = []

    iterable = x_T.split(split_size=max_batch, dim=0)    # type: ignore
    pbar = tqdm(iterable, desc="Sampling")    # type: ignore

    # injecting the param into the ode wrapper
    ode_fn.pbar = pbar    # type: ignore

    # initialize the space (integrating backwards from T to 0)
    with torch.no_grad():
        if method == 'dopri5':
            for chunk in pbar:
                # even though rk45 is an adaptive timestepping solver, it will stay on the gridpoints defined by linspace.
                # so just pass a super big number as the amount of steps to make the grid fine enough
                t_vals = torch.linspace(T, 0., 10_000, device=device)
                sol: torch.Tensor = odeint(ode_fn, chunk, t_vals, method=method, atol=a_tol, rtol=r_tol)    # type: ignore
                traj.append(sol)    # type: ignore

        else:    # method in ['euler', 'midpoint']
            for chunk in pbar:
                t_vals = torch.linspace(T, 0., num_steps, device=device)
                sol: torch.Tensor = odeint(ode_fn, chunk, t_vals, method=method)    # type: ignore
                traj.append(sol)    # type: ignore

    # torchdiffeq returns shape (time_steps, batch_size, ...)
    return torch.cat(traj, dim=1)

@torch.inference_mode()
def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider, sampler: TorchKacConstantSampler | None, save_path: str) -> None:
    device = next(model.parameters()).device

    # Assert that a properly inititialized sampler has been passed
    assert isinstance(sampler, TorchKacConstantSampler)

    # Initiate Kac noise
    t = torch.ones(args.num_samples, 1, device=device) * args.T
    x_T = sampler.sample(t.squeeze(1), dim=data.data_dims.total_dimension).to(device)
    
    # Map command-line args to torchdiffeq methods
    method_map = {
        'ee': 'euler',
        'rk2': 'midpoint',
        'rk45': 'dopri5'}
    solver_method = method_map[args.sampler_kac]
    
    print(f"Sampling {args.num_samples} images using torchdiffeq Kac Flow solver: {solver_method}...")
    
    full_traj = sample_ode(
        model=model,
        data = data,
        x_T=x_T,
        T=1.0,
        num_steps=args.num_teacher_steps,
        device=device,
        method=solver_method,
        max_batch=args.num_samples,
        a_tol = args.abs_tol,
        r_tol = args.rel_tol
    )
    
    # Extract the final step (t=0)
    samples = full_traj[-1].view(args.num_samples, data.data_dims.channels, data.data_dims.width, data.data_dims.height)
    
    print(f"Generated samples shape: {samples.shape}")

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