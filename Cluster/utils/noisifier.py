"""For now this primarily collects the forward noising processes for later use in the distillation algorithm."""

import argparse

import torch

class Noisify():

    def __init__(self, args: argparse.Namespace):

        self.args = args

        if self.args.which == 'diffusion':
            self.noisify = self.diffusion

        elif self.args.which == 'kac':
            self.noisify = self.kac

        else:    # self.args.which == 'mmd':
            self.noisify = self.mmd


    def diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # * closed form solution to the reverse time SDE formulation of the diffusion process, see e.g. "Song et al 2021 - Score based generative modeling through SDEs"

        from Cluster.utils.diffusion import Diffusion

        b_t = Diffusion.b(t).view(-1, 1, 1, 1)
        return torch.sqrt(1 - b_t**2) * torch.randn_like(x0) + b_t * x0
    

    def kac(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # * Mean-Reverting Forward Process see 'Duong Chemseddine 2025 - Telegraphers Generative Model via Damped Wave Equations'
        
        from Cluster.utils.kac import Kac
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        shape = x0.shape
        
        # time parameterization for the noise
        t_noise: torch.Tensor = Kac.g(t, self.args.T, self.args.kac_g)

        # Broadcast 1D t (e.g., batch_size) to target shape (e.g., [batch_size, C, H, W])
        t_broadcast = t_noise.view(-1, 1, 1, 1)

        # Determine max jumps needed (Mean + 5 Standard Deviations buffer)
        t_max = torch.max(t_noise)
        max_events = int((self.args.kac_c * t_max) + 5 * torch.sqrt(self.args.kac_c * t_max) + 10)

        # Generate inter-arrival times for all dimensions simultaneously
        # Shape: [max_events, *shape]
        inter_arrivals = torch.empty((max_events, *shape), device=device).exponential_(1.0 / self.args.kac_c)

        # Convert to absolute arrival times
        arrival_times = torch.cumsum(inter_arrivals, dim=0)

        # Clamp times to maximum t. 
        # Any event occurring after t is clamped to t, making its subsequent duration 0.
        clamped_times = torch.clamp(arrival_times, max=t_broadcast)

        # Prepend t=0 to calculate durations
        zeros = torch.zeros((1, *shape), device=device)
        full_times = torch.cat([zeros, clamped_times], dim=0)

        # Calculate time spent in each state (dt)
        durations = full_times[1:] - full_times[:-1]

        # Create alternating signs for the integrand (+1, -1, +1, -1...)
        signs = torch.ones((max_events, *shape), device=device)
        signs[1::2] = -1.0

        # Sample initial random directions D_0 in {-1, 1}
        initial_directions = torch.randint(0, 2, shape, device=device).float() * 2.0 - 1.0

        # Integrate: sum of (duration * sign) * velocity * initial_direction
        integral = torch.sum(durations * signs, dim=0)
        noise = initial_directions * self.args.kac_c * integral
        
        # Calculate the mean reverting kac process starting in x0
        f_t = Kac.f(t, self.args.T, self.args.kac_f).view(-1, 1, 1, 1)

        # use that to corrupt the original sample fully
        return f_t * x0 + noise
    

    def mmd(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        from Cluster.utils.mmd import MMD

        # data schedule
        f_t = MMD.f(t=t)

        # noise schedule
        g_t = MMD.g(t=t)

        # noise at gt
        _, noise = MMD.get_noise(t=g_t, x=x0, b=self.args.mmd_b)

        # use that to corrupt the original sample fully
        return f_t * x0 + noise