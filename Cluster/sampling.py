import argparse
import os

import torch
from torchvision.utils import make_grid, save_image    # type: ignore

import matplotlib.pyplot as plt

from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.reversals import Reversal


@torch.inference_mode()
def sample(args: argparse.Namespace, model: torch.nn.Module,
                    data: DataProvider, reversal_fns: Reversal,
                    sampler: TorchKacConstantSampler | None) -> torch.Tensor:

    batch_size = args.sampling_batch_size
    device = next(model.parameters()).device

    all_samples = []

    for i in range(0, args.sampling_num_samples, batch_size):
        curr_batch_size = min(batch_size, args.sampling_num_samples - i)
        if i % 200 == 0:
            print(f'curr batch:  {i}')
        
        # Initialize with random noise
        if args.which == 'mmd':
            from Cluster.utils.mmd import MMD
            _, x_batch = MMD.get_noise(t=torch.ones(curr_batch_size) * args.T,
                                    x=torch.ones(curr_batch_size, data.data_dims.channels, data.data_dims.width, data.data_dims.height),
                                    b=args.mmd_b)

        elif args.which == 'kac':
            assert isinstance(sampler, TorchKacConstantSampler)
            x_batch = sampler.sample(torch.ones(curr_batch_size, 1, device=device) * args.T, dim=data.data_dims.total_dimension).to(device)
            x_batch = x_batch.view(curr_batch_size, data.data_dims.channels, data.data_dims.width, data.data_dims.height)

        else:    # args.which == 'diffusion':
            x_batch = torch.randn((curr_batch_size, data.data_dims.channels, data.data_dims.width, data.data_dims.height), device=device)


        # Properly scale continuous time from T down to epsilon
        # ! Diffusion and MMD sample until 1e-5 due to their singularity,
        # ! Kac samples until 0
        time_steps = torch.linspace(args.T, args.time_truncation, args.sampling_num_steps, device=device)
        dt = (args.T - args.time_truncation) / args.sampling_num_steps

        for step_idx, t_val in enumerate(time_steps):
            
            if step_idx % 100 == 0 or len(time_steps) - step_idx <= 20:
                print(f"Step {step_idx}/{args.sampling_num_steps}", end='\r')

            # Broadcast the continuous time value to the batch size
            t = torch.ones(curr_batch_size, device=device) * t_val
            
            if args.sampling_sampler == 'em':
                # Don't add random noise at the very last step
                noise_injection_bool = (step_idx != args.sampling_num_steps - 1)
                    
                x_batch = reversal_fns.integrator(
                    model=model,
                    x_batch=x_batch,
                    t_start=t,
                    dt=dt,
                    num_substeps=1,
                    noise_injection_bool=noise_injection_bool
                )

            else:
                x_batch = reversal_fns.integrator(
                    model=model,
                    x_batch=x_batch,
                    t_start=t,
                    dt=dt,
                    num_substeps=1
                )
        
        all_samples.append(x_batch.cpu())

    return torch.cat(all_samples, dim=0)


def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider, sampler: TorchKacConstantSampler | None,
                   reversal_fns: Reversal, save_path: str):
    """
    the sampler argument only exists so that the full wrapper does not show a warning for a missing argument
    """


    samples = sample(args=args, model=model, data=data, reversal_fns=reversal_fns, sampler=sampler)

    # if your images are normalized to [-1, 1], rescale to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    if args.sampling_mode == '8x8':
        grid = make_grid(samples, nrow=8, padding=2, normalize=False)

        plt.figure(figsize=(8, 8))    # type: ignore    plt badly typed
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)    # type: ignore    plt badly typed
        plt.axis("off")    # type: ignore    plt badly typed

        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)    # type: ignore    plt badly typed
        
        plt.close()
    else:    # args.sampling_sampler_mode == 'set'
        os.makedirs(save_path, exist_ok=True)
        
        # save_image automatically handles the C, H, W shape and normalizes to 0-255 internally
        for i, img in enumerate(samples):
            img_path = os.path.join(save_path, f"sample_{i:05d}.png")
            save_image(img, img_path)