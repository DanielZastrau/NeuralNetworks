import argparse
import torch
from torchdiffeq import odeint    # type: ignore
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid    # type: ignore

from utils.sample_kac import TorchKacConstantSampler

_IMG_SIZE = 32

class ODEWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.nfe = 0
        self.pbar = None

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        self.nfe += 1
        if self.pbar is not None:
            self.pbar.set_postfix({"t": f"{t.item():.4f}", "NFE": self.nfe})

        B = x.shape[0]
        x_img = x.view(B, 3, _IMG_SIZE, _IMG_SIZE)
        t_vec = torch.full((B,), float(t), device=x.device)
        
        v = self.model(x_img, t_vec)
            
        return v.view(x.shape)

def sample_ode(model: torch.nn.Module, x_T: torch.Tensor, T: float, num_steps: int,
               device: torch.device, method: str = 'euler', max_batch: int = 512):
    """
    args:
        model: torch.nn.module - the neural network
        x_T: torch.Tensor - tensor of samples of the noise distribution
    """
    # initialize the ode fn
    ode_fn = ODEWrapper(model).to(device)

    # initialize the space (integrating backwards from T to 0)
    t_vals = torch.linspace(T, 0., num_steps, device=device)
    
    # track solution paths for later use
    traj: list[torch.Tensor] = []

    with torch.no_grad():
        iterable = x_T.split(split_size=max_batch, dim=0)    # type: ignore
        pbar = tqdm(iterable, desc="Sampling")    # type: ignore
        
        # injecting the param into the ode wrapper
        ode_fn.pbar = pbar    # type: ignore

        for chunk in pbar:
            sol: torch.Tensor = odeint(ode_fn, chunk, t_vals, method=method)    # type: ignore
            traj.append(sol)    # type: ignore

    # torchdiffeq returns shape (time_steps, batch_size, ...)
    return torch.cat(traj, dim=1)

@torch.inference_mode()
def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, sampler: TorchKacConstantSampler) -> None:
    device = next(model.parameters()).device
    batch_size = 64
    
    # Initiate Kac noise
    t = torch.ones(batch_size, 1, device=device) * args.T
    x_T = sampler.sample(t.squeeze(1), dim=3*32*32).to(device)
    
    # Map command-line args to torchdiffeq methods
    method_map = {
        'ee': 'euler',
        'rk2': 'midpoint',
        'rk45': 'dopri5'}
    solver_method = method_map[args.sampler_kac]
    
    print(f"Sampling {batch_size} images using torchdiffeq Kac Flow solver: {solver_method}...")
    
    full_traj = sample_ode(
        model=model,
        x_T=x_T,
        T=1.0,
        num_steps=args.num_steps,
        device=device,
        method=solver_method,
        max_batch=batch_size,
    )
    
    # Extract the final step (t=0)
    samples = full_traj[-1].view(batch_size, 3, 32, 32)
    
    print(f"Generated samples shape: {samples.shape}")

    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    grid = make_grid(samples, nrow=8, padding=8, normalize=False)

    plt.figure(figsize=(4, 4))    # type: ignore
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)    # type: ignore
    plt.axis("off")    # type: ignore

    path_to_save = f"./{args.where}_{args.which}_{args.epochs}_samples_8x8_fODE_{args.sampler_kac}.png"
    if args.where == 'cluster': 
        path_to_save = f"/homes/math/zastrau/NeuralNetworkSamples/{path_to_save}"
        
    plt.savefig(path_to_save, dpi=200, bbox_inches="tight", pad_inches=0)    # type: ignore
    print(f"Saved generated samples to {path_to_save}")
    plt.close()