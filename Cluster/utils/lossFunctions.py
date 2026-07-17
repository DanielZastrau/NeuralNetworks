import argparse

import torch

from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.dataHandling import DataProvider

class LossFns():
    """Provides the loss functions such that the training framework can stay the same, currently only used in the training module"""


    def __init__(self, args: argparse.Namespace, sampler: TorchKacConstantSampler | None,
                 data: DataProvider):
        self.args = args
        self.sampler = sampler
        self.data = data

        if args.which == 'diffusion':
            self.loss = self.diffusion

        elif args.which == 'kac':
            self.loss = self.kac

        elif args.which == 'mmd':
            self.loss = self.mmd

        elif args.which == 'schrödinger':
            self.loss = self.schrödinger
            

    def diffusion(self, model: torch.nn.Module, xbatch: torch.Tensor) -> torch.Tensor:
        from Cluster.utils.diffusion import Diffusion

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0 = xbatch

        # sample time steps t uniformly from [eps, 1)
        t = torch.rand(xbatch.size(0), device=device)
        t = torch.clamp(t, min=self.args.time_truncation, max=1)

        # draw standard gaussian noise matching the shape of x0
        noise = torch.randn_like(x0)
        b_t = Diffusion.b(t).view(-1, 1, 1, 1)

        # compute x
        x_corrupted = b_t * x0 + torch.sqrt(1 - b_t**2) * noise

        # simplyfiy the target
        # target = - noise / torch.sqrt(1 - b_t**2)

        # simplyfiy the target even further and only learn the noise and scale 
        # to the score later during inference
        target = noise

        # evaluate the neural network score prediction
        # ! scale the time up to [0, 1000.0] since the unet time embedding
        # ! expects this time scale otherwise the time embedding will basically
        # ! look the same at t=0.01 and t=0.99
        pred = model(x_corrupted, t * 1000.0)

        # compute empirical loss
        loss = torch.nn.functional.mse_loss(pred.float(), target)

        if self.args.training_verbosity == 'verbose' and not torch.isfinite(loss):
            if not torch.isfinite(pred).all():
                print("NaN/Inf detected in model predictions.")
            if not torch.isfinite(target).all():
                print("NaN/Inf detected in target tensor.")

        return loss
    
    def schrödinger(self, model: torch.nn.Module, xbatch: torch.Tensor) -> torch.Tensor:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x0 = xbatch

        t = torch.rand(xbatch.size(0), device=device)
        t = torch.clamp(t, min=self.args.time_truncation, max=1)
        noise = torch.randn_like(x0)

        x_corrupted = (1 - t) * x0 + torch.sqrt(t) * noise

        # * See my masters thesis
        # ? this is the eulerian formulation
        # eulerian_target = 0.5 * ( - x0 - ((x0 - x_corrupted) / t) )

        # ? this is the lagrangian formulation, we prefer that one because it is more stable near 0
        lagrangian_target = -x0 + 0.5 * torch.sqrt(t)**(-1) * noise

        # ! scale the time up to [0, 1000.0] somce the unet time embedding expects this time scale otherwise
        # ! the embedding will essentially look the same for all t in the time-interval
        pred = model(x_corrupted, t * 1000.0)

        loss = torch.nn.functional.mse_loss(pred.float(), lagrangian_target)

        return loss
    
    def kac(self, model: torch.nn.Module, xbatch: torch.Tensor) -> torch.Tensor:
        from Cluster.utils.velo_utils import compute_velocity
        from Cluster.utils.kac import Kac

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert isinstance(self.sampler, TorchKacConstantSampler)

        B = xbatch.size(0)
        x0 = xbatch.view(B, -1).to(device)

        # Sample original time t
        t = torch.rand(B, 1, device=device) * self.args.T

        # data schedule
        f_t = Kac.f(t.squeeze(1), self.args.T, self.args.kac_f).unsqueeze(1)
        df_t = Kac.df(t.squeeze(1), self.args.T, self.args.kac_f).unsqueeze(1)

        # noise schedule
        g_t = Kac.g(t.squeeze(1), self.args.T, self.args.kac_g)
        dg_t = Kac.dg(t, self.args.T, self.args.kac_g)
        
        # compute x corrupted -----
        # Sample noise using g(t)
        tau: torch.Tensor = self.sampler.sample(g_t, dim=self.data.data_dims.total_dimension).to(device)

        # use it to corrupt x0 fully, according to the mean reverting process
        x_corrupted = f_t * x0 + tau

        # compute the target
        drift = df_t * x0

        with torch.no_grad():
            # Compute velocity using g(t)
            velo = dg_t * compute_velocity((x_corrupted - f_t * x0), g_t.unsqueeze(1), a=self.args.kac_a, c=self.args.kac_c, epsilon=1e-4, T=self.args.T)

        # Model is conditioned on original time t
        pred = model(x_corrupted.view(
            B,
            self.data.data_dims.channels,
            self.data.data_dims.height,
            self.data.data_dims.width
        ), t.squeeze(1) * 1000.0).view(B, -1)

        loss = torch.nn.functional.mse_loss(pred.float(), velo + drift)

        if self.args.training_verbosity == 'verbose' and not torch.isfinite(loss):
            if not torch.isfinite(pred).all():
                print("NaN/Inf detected in model predictions.")
            if not torch.isfinite(velo + drift).all():
                print("NaN/Inf detected in target tensor.")

        return loss
    

    def mmd(self, model: torch.nn.Module, xbatch: torch.Tensor) -> torch.Tensor:
        from Cluster.utils.mmd import MMD

        x0 = xbatch

        # sample randomly uniformly from [0, 1]
        t = torch.rand(x0.size(0), device=x0.device) * self.args.T

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
        pred = model(x_corrupted, t * 1000.0)

        loss = torch.nn.functional.mse_loss(pred.float(), target)

        if self.args.training_verbosity == 'verbose' and not torch.isfinite(loss):
            if not torch.isfinite(pred).all():
                print("NaN/Inf detected in model predictions.")
            if not torch.isfinite(target).all():
                print("NaN/Inf detected in target tensor.")

        return loss