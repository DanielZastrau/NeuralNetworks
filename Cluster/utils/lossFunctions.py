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

        elif args.which == 'kac':
            self.loss = self.kac

        else:    # args.which == 'mmd':
            self.loss = self.mmd
            

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

        # data schedule
        f_t = Kac.f(t, self.args.T, self.args.kac_f)
        df_t = Kac.df(t, self.args.T, self.args.kac_f)

        # noise schedule
        g_t = Kac.g(t, self.args.T, self.args.kac_g)
        dg_t = Kac.dg(t, self.args.T, self.args.kac_g)
        
        # compute x corrupted -----
        # Sample noise using g(t)
        tau: torch.Tensor = self.sampler.sample(g_t, dim=3 * 32 * 32).to(device).view(B, -1)

        # use it to corrupt x0 fully, according to the mean reverting process
        x_corrupted = f_t * x_0 + tau

        # compute the target
        drift = df_t * x_0

        with torch.no_grad():
            # Compute velocity using g(t)
            velo = dg_t * compute_velocity((x_corrupted - f_t * x_0), g_t, a=self.args.a, c=self.args.c, epsilon=1e-4, T=self.args.T)

        # Model is conditioned on original time t
        pred = model(x_corrupted.view(B, 3, 32, 32), t.squeeze(1)).view(B, -1)
        loss = torch.nn.functional.mse_loss(pred, velo + drift)

        return loss
    

    def mmd(self, model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:

        from Cluster.utils.mmd import MMD

        x0 = mini_batch

        # sample randomly uniformly from [0, 1]
        # TODO add normalization for larger interval like for the other processes
        t = torch.rand(x0.size(0), device=x0.device)

        # data schedule
        f_t = MMD.f(t=t).view(-1, 1, 1, 1)
        df_t = MMD.df(t=t).view(-1, 1, 1, 1)

        # noise schedule
        g_t = MMD.g(t=t).view(-1, 1, 1, 1)
        dg_t = MMD.dg(t=t).view(-1, 1, 1, 1)

        # noise at gt
        pre_noise, noise = MMD.get_noise(t=g_t, x=x0, b=self.args.mmd_b)

        # use that to corrupt the original sample fully, according to the mean reverting process
        x_corrupted = f_t * x0 + noise

        # compute the target and simplify the target
        # target = df_t * x0 + dg_t * ((x_corrupted - f_t * x0)/(self.args.mmd_b * (torch.exp(g_t / self.args.mmd_b) - 1)))
        # target = df_t * x0 + dg_t * (noise/(self.args.mmd_b * (torch.exp(g_t / self.args.mmd_b) - 1)))
        target = df_t * x0 + dg_t * pre_noise / torch.exp(g_t / self.args.mmd_b)

        # compute the mse loss
        pred = model(x_corrupted, t)
        loss = torch.nn.functional.mse_loss(pred, target)

        return loss