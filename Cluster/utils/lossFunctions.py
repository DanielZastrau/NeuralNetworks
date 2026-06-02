import argparse

import torch

from Cluster.utils.sample_kac import TorchKacConstantSampler

class LossFns():
    """Provides the loss functions such that the training framework can stay the same, currently only used in the training module"""


    def __init__(self, args: argparse.Namespace, sampler: TorchKacConstantSampler | None):
        self.args = args
        self.sampler = sampler

        if args.which == 'diffusion':
            self.loss = self.diffusion

        else:    # args.which == 'kac'
            self.loss = self.kac


    def diffusion(self, model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:
        from Cluster.utils.diffusion import b

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        x_0 = mini_batch

        # sample time steps t uniformly from [eps, 1)
        t = torch.rand(mini_batch.size(0), device=device)
        t = torch.clamp(t, min=self.args.time_truncation, max=1)

        # draw standard gaussian noise matching the shape of x_0
        noise = torch.randn_like(x_0)

        # compute b(t)
        b_t = b(t)
        b_t = b_t.view(-1, 1, 1, 1)  # Reshape b_t to (batch_size, 1, 1, 1) for broadcasting

        # compute x
        x_corrupted = b_t * x_0 + torch.sqrt(1 - b_t**2) * noise

        # simplyfiy the target
        # target = - noise / torch.sqrt(1 - b_t**2)

        # simplyfiy the target even further and only learn the noise and scale to the score later during inference
        target = noise

        # evaluate the neural network score prediction
        pred = model(x_corrupted, t)

        # compute empirical loss
        loss = torch.nn.functional.mse_loss(pred, target)

        return loss
    
    def kac(self, model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:
        from Cluster.utils.velo_utils import compute_velocity
        from Cluster.utils.kac import Kac

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert isinstance(self.sampler, TorchKacConstantSampler)

        B = mini_batch.size(0)
        x_0 = mini_batch.view(B, -1).to(device)

        # Sample original time t
        t = torch.rand(B, 1, device=device) * self.args.T

        # --- Use g(t) for noise process time ---
        t_for_noise_proc = Kac.g(t.squeeze(1), self.args.T, self.args.g)
        
        # Sample noise using g(t)
        tau: torch.Tensor = self.sampler.sample(t_for_noise_proc, dim=3 * 32 * 32).to(device)
        
        # Use original t for signal schedule f(t)
        f_t: torch.Tensor = Kac.f(t.squeeze(1), self.args.T, self.args.f).unsqueeze(1)

        x_t = f_t * x_0 + tau
        drift = Kac.df(t.squeeze(1), self.args.T, self.args.f).unsqueeze(1) * x_0

        with torch.no_grad():
            # Compute velocity using g(t)
            velo = Kac.dg(t, self.args.T, self.args.g) * \
                compute_velocity((x_t - f_t * x_0), t_for_noise_proc.unsqueeze(1), a = self.args.a, c = self.args.c, epsilon=1e-4, T = self.args.T)

        # Model is conditioned on original time t
        pred = model(x_t.view(B, 3, 32, 32), t.squeeze(1)).view(B, -1)
        loss = torch.nn.functional.mse_loss(pred, velo + drift)

        return loss
    

    def mmd(self, model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:

        from Cluster.utils.mmd import MMD

        x0 = mini_batch

        # sample randomly uniformly from [0, 1]
        # TODO add normalization for larger interval like for the other processes
        t = torch.rand_like(x0, device=x0.device)

        # data schedule
        f_t = MMD.f(t=t)
        df_t = MMD.df(t=t)

        # noise schedule
        g_t = MMD.g(t=t)
        dg_t = MMD.dg(t=t)

        # compute x corrupted
        # TODO for this we need noise sampled at time gt
        x_corrupted = f_t * x0 + 

        # compute the target
        target = df_t * x0 + dg_t * (x-f_t/)