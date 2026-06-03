import argparse
import os

import torch
from torchvision.utils import make_grid, save_image    # type: ignore

import matplotlib.pyplot as plt

from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.reversals import Reversal

@torch.inference_mode()
def sample_diff(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider, reversal_fns: Reversal,
                mode: str = 'teacher') -> torch.Tensor:
    print(f"Sampling {args.num_samples} images using {args.sampler_diff} sampling...")

    batch_size = args.sampling_batch_size
    device = next(model.parameters()).device

    all_samples = []

    for i in range(0, args.num_samples, batch_size):
        curr_batch_size = min(batch_size, args.num_samples - i)
        print(f'curr batch size:  {curr_batch_size}')

        # Initialize x with random noise from the prior distribution
        x_batch = torch.randn((curr_batch_size, data.data_dims.channels, data.data_dims.width, data.data_dims.height), device=device)

        # ! adaptive timestepping is only allowed for the teacher model right now
        if args.sampler_diff == 'pfode' and mode == 'teacher':
            x_batch = reversal_fns.rk45_wrapper(
                model=model,
                data=data,
                x_batch=x_batch,
                t_start=args.T,
                t_end=args.time_truncation
            )
        
        elif mode == 'student' or args.sampler_diff == 'sde':

            # Properly scale continuous time from T down to epsilon
            time_steps = torch.linspace(1.0, args.time_truncation, args.num_steps, device=device)
            dt = (1.0 - args.time_truncation) / args.num_steps

            for step_idx, t_val in enumerate(time_steps):

                if step_idx % 100 == 0 or len(time_steps) - step_idx <= 20:
                    print(f"Step {step_idx}/{args.num_steps}")

                # Broadcast the continuous time value to the batch size
                t = torch.ones(curr_batch_size, device=device) * t_val

                # Don't add random noise at the very last step
                noise_injection_bool = (step_idx != args.num_steps - 1)
                    
                x_batch = reversal_fns.diffusion_euler_maruyama(
                    model=model,
                    x_batch=x_batch,
                    t_start=t,
                    dt=dt,
                    num_substeps=1,
                    noise_injection_bool=noise_injection_bool
                )


        all_samples.append(x_batch.cpu())

    return torch.cat(all_samples, dim=0)


@torch.inference_mode()
def sample_kac(args: argparse.Namespace, model: torch.nn.Module,
                    data: DataProvider, reversal_fns: Reversal,
                    sampler: TorchKacConstantSampler, mode: str = 'teacher') -> torch.Tensor:
    print(f"Sampling {args.num_samples} images using {args.sampler_kac} sampling...")

    batch_size = args.sampling_batch_size
    device = next(model.parameters()).device

    all_samples = []

    for i in range(0, args.num_samples, batch_size):
        curr_batch_size = min(batch_size, args.num_samples - i)

        # Initiate Kac noise
        x_batch = sampler.sample(torch.ones(curr_batch_size, 1, device=device) * args.T, dim=data.data_dims.total_dimension).to(device)
        x_batch = x_batch.view(curr_batch_size, data.data_dims.channels, data.data_dims.width, data.data_dims.height)

        if args.sampler_kac == 'rk45' and mode == 'teacher':
            x_batch = reversal_fns.rk45_wrapper(
                model=model, 
                data=data, 
                x_batch=x_batch, 
                t_start=args.T,
                t_end=0
            )

        else:    # args.sampler_kac in ['ee', 'rk2'] or mode == 'student'
            # ! Kac has finite dynamics therefore we can integrate all the way to 0
            if args.sampler_kac == 'ee' or mode == 'student':
                reversal_fn = reversal_fns.explicit_euler
            else:    # args.sampler_kac == 'rk2'
                reversal_fn = reversal_fns.rk2

            # Properly scale continuous time from T down to epsilon
            time_steps = torch.linspace(args.T, 0, args.num_steps, device=device)
            dt = args.T / args.num_steps

            for step_idx, t_val in enumerate(time_steps):
                
                if step_idx % 100 == 0 or len(time_steps) - step_idx <= 20:
                    print(f"Step {step_idx}/{args.num_steps}")

                # Broadcast the continuous time value to the batch size
                t = torch.ones(curr_batch_size, device=device) * t_val
                
                x_batch = reversal_fn(
                    model=model,
                    x_batch=x_batch,
                    t_start=t,
                    dt=dt,
                    num_substeps=1
                )
            
        all_samples.append(x_batch)

    return torch.cat(all_samples, dim=0)


@torch.inference_mode()
def sample_mmd(args: argparse.Namespace, model: torch.nn.Module,
                    data: DataProvider, reversal_fns: Reversal, mode: str = 'teacher') -> torch.Tensor:
    from Cluster.utils.mmd import MMD

    print(f"Sampling {args.num_samples} images using {args.sampler_mmd} sampling...")

    batch_size = args.sampling_batch_size
    device = next(model.parameters()).device

    all_samples = []

    for i in range(0, args.num_samples, batch_size):
        curr_batch_size = min(batch_size, args.num_samples - i)

        # Initialize with random noise
        _, x_batch = MMD.get_noise(t=torch.ones(curr_batch_size) * args.T,
                                x=torch.ones(curr_batch_size, data.data_dims.channels, data.data_dims.width, data.data_dims.height),
                                b=args.mmd_b)


        if args.sampler_mmd == 'ee':
            reversal_fn = reversal_fns.explicit_euler
        else:    # args.sampler_kac == 'rk2':
            reversal_fn = reversal_fns.rk2

        # Properly scale continuous time from T down to epsilon
        # ! MMD has finite dynamics therefore we can integrate all the way to 0
        time_steps = torch.linspace(args.T, 0, args.num_steps, device=device)
        dt = args.T / args.num_steps

        for step_idx, t_val in enumerate(time_steps):
            
            if step_idx % 100 == 0 or len(time_steps) - step_idx <= 20:
                print(f"Step {step_idx}/{args.num_steps}")

            # Broadcast the continuous time value to the batch size
            t = torch.ones(curr_batch_size, device=device) * t_val
            
            x_batch = reversal_fn(
                model=model,
                x_batch=x_batch,
                t_start=t,
                dt=dt,
                num_substeps=1
            )
        
        all_samples.append(x_batch)

    return torch.cat(all_samples, dim=0)


def sample_wrapper(args: argparse.Namespace, model: torch.nn.Module, data: DataProvider, sampler: TorchKacConstantSampler | None,
                   reversal_fns: Reversal, save_path: str, mode: str = 'teacher'):
    """
    the sampler argument only exists so that the full wrapper does not show a warning for a missing argument
    """

    if args.which == 'diffusion':
        samples = sample_diff(args=args, model=model, data=data, reversal_fns=reversal_fns, mode=mode)
    elif args.which == 'kac':
        # Assert that a properly inititialized sampler has been passed
        assert isinstance(sampler, TorchKacConstantSampler)
        samples = sample_kac(args=args, model=model, data=data, reversal_fns=reversal_fns, sampler=sampler, mode=mode)
    else:    # args.which == 'mmd':
        samples = sample_mmd(args=args, model=model, data=data, reversal_fns=reversal_fns, mode=mode)

    print(f"Generated samples shape: {samples.shape}")  # Should be (64, 3, 32, 32)

    # if your images are normalized to [-1, 1], rescale to [0, 1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    if args.sampler_mode == '8x8':
        grid = make_grid(samples, nrow=8, padding=2, normalize=False)

        plt.figure(figsize=(8, 8))    # type: ignore    plt badly typed
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)    # type: ignore    plt badly typed
        plt.axis("off")    # type: ignore    plt badly typed

        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)    # type: ignore    plt badly typed
        print(f"Saved generated samples to {save_path}")
        
        plt.close()
    else:    # args.sampler_diff_mode == 'set'
        os.makedirs(save_path, exist_ok=True)
        
        # save_image automatically handles the C, H, W shape and normalizes to 0-255 internally
        for i, img in enumerate(samples):
            img_path = os.path.join(save_path, f"sample_{i:05d}.png")
            save_image(img, img_path)
            
        print(f"Saved {len(samples)} individual images to {save_path}/")