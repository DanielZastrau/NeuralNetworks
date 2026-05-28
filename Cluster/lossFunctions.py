import argparse

import torch

from utils.sample_kac import TorchKacConstantSampler

class LossFns():

    def __init__(self, args: argparse.Namespace, sampler: TorchKacConstantSampler | None):
        self.args = args
        self.sampler = sampler

        if args.which == 'diffusion':
            self.loss = self.diffusion

        elif args.which == 'kac':
            self.loss = self.kac

    def diffusion(self, model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:
        from utils.diffusion import b

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        x_0 = mini_batch

        # sample time steps t uniformly from [0, 1)
        t = torch.rand(mini_batch.size(0), device=device)

        # draw standard gaussian noise matching the shape of x_0
        noise = torch.randn_like(x_0)

        # compute b(t)
        b_t = b(t)
        b_t = b_t.view(-1, 1, 1, 1)  # Reshape b_t to (batch_size, 1, 1, 1) for broadcasting

        # compute x
        x = b_t * x_0 + torch.sqrt(1 - b_t**2) * noise

        # simplyfiy the target
        # target = - noise / torch.sqrt(1 - b_t**2)

        # simplyfiy the target even further and only learn the noise and scale to the score later during inference
        target = noise

        # evaluate the neural network score prediction
        pred = model(x, t)

        # compute empirical loss
        loss = torch.nn.functional.mse_loss(pred, target)

        return loss
    
    def kac(self, model: torch.nn.Module, mini_batch: torch.Tensor) -> torch.Tensor:
        from utils.velo_utils import compute_velocity
        from utils.kac import f, df, g, dg

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert isinstance(self.sampler, TorchKacConstantSampler)

        B = mini_batch.size(0)
        x_0 = mini_batch.view(B, -1).to(device)

        # Sample original time t
        t = torch.rand(B, 1, device=device) * self.args.T

        # --- Use g(t) for noise process time ---
        t_for_noise_proc = g(t.squeeze(1), self.args.T)
        
        # Sample noise using g(t)
        tau: torch.Tensor = self.sampler.sample(t_for_noise_proc, dim=3 * 32 * 32).to(device)
        
        # Use original t for signal schedule f(t)
        f_t: torch.Tensor = f(t.squeeze(1), self.args.T).unsqueeze(1)

        x_t = f_t * x_0 + tau
        drift = df(t.squeeze(1), self.args.T).unsqueeze(1) * x_0

        with torch.no_grad():
            # Compute velocity using g(t)
            velo = dg(t, self.args.T) * \
                compute_velocity((x_t - f_t * x_0), t_for_noise_proc.unsqueeze(1), a = self.args.a, c = self.args.c, epsilon=1e-4, T = self.args.T)

        # Model is conditioned on original time t
        pred = model(x_t.view(B, 3, 32, 32), t.squeeze(1)).view(B, -1)
        loss = torch.nn.functional.mse_loss(pred, velo + drift)

        return loss