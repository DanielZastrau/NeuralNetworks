import os
import argparse

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class Diffusion():
    """Implements the continuous ddpm model as described in 2021 - Song et al - Score based generative modeling through SDEs
    The prediction target is given by Eq. (7) and we use the probability flow ODE for sampling with a final application of Tweedie's formula."""

    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.beta0 = 0.1
        self.beta1 = 20

        self.T = 1_024

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000)
        )

        self.base = '/work/zastrau/SongCont'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.base = '/work/zastrau/SongCont/DDPM'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0

    def beta(self, t: torch.Tensor) -> torch.Tensor:

        return self.beta0 + t * (self.beta1 - self.beta0)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        return - 0.5 * self.beta(t).view(-1, *([1] * (x.dim() - 1))) * x

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        return torch.sqrt(self.beta(t)).view(-1, *([1] * (x.dim() - 1)))

    def b(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        return torch.exp(- 0.5 * (t * self.beta(torch.zeros(1)) + 0.5 * t**2 * (self.beta(torch.ones(1)) - self.beta(torch.zeros(1))))).view(-1, *([1] * (x.dim() - 1)))

    def v(self, t: torch.Tensor, x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:

        with torch.no_grad():
            pred_score = model(x, t)

        return self.f(t, x) - 0.5 * self.g(t, x)**2 * pred_score


    def get_model(self):

        #! the network natively implements group normalization
        return UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=128, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=4, attention_resolutions=(2,),
                         dropout=0.1, use_scale_shift_norm=True,
                         resblock_updown=True, use_new_attention_order=True,
                         use_rff=True, rff_scale=16.0).to(self.device)

    def get_ema(self, model: torch.nn.Module):

        return torch.optim.swa_utils.AveragedModel(model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999))

    def get_optim(self, model: torch.nn.Module):

        return torch.optim.Adam(model.parameters(), lr=2e-4)

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        t = torch.rand((x0.shape[0],), device=self.device)
        t = torch.clamp(t, min=1e-5)

        z = torch.randn_like(x0)

        mean = self.b(t, x0)
        variance_sq = 1 - mean**2
        variance = torch.sqrt(variance_sq)

        xt = mean * x0 + variance * z
        
        pred_score = model(xt, t)
        target_score = -z / variance

        # The dimensionality denominator is irrelevant for the minimum, it gets absorbed by the propto symbol in the paper
        weight = variance_sq

        mse_loss = torch.nn.functional.mse_loss(pred_score, target_score, reduction='none')
        return (weight * mse_loss).view(x0.shape[0], -1).sum(dim=1).mean()

    def sample(self, model: torch.nn.Module, amount: int):

        samples = []

        for i in range((amount // 512) + 1):
            how_many = min(512, amount - i * 512)

            xT = torch.randn((how_many,
                              self.data.data_dims.channels,
                              self.data.data_dims.height,
                              self.data.data_dims.width),
                              device=self.device,
                              dtype=torch.float32)

            dt = (1 - 1e-3) / self.T
            xt = xT
            for j in range(self.T):

                t = 1 - j * dt
                t_tensor = torch.full((xt.shape[0],), t, dtype=torch.float32, device=xt.device)

                xt = xt - self.v(t_tensor, xt, model) * dt

            # Final Tweedies application
            final_time = torch.full((xt.shape[0],), 1e-3, dtype=torch.float32, device=xt.device)
            with torch.no_grad():
                final_score_pred = model(xt, final_time)
            xt = (xt + (1 - self.b(final_time, xt)**2) * final_score_pred ) / self.b(final_time, xt)

            samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    def train(self):

        model = self.get_model()
        ema = self.get_ema(model=model)
        optim = self.get_optim(model=model)

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for iteration in range(1_000_000):
            if (iteration + 1) % 1_000 == 0:
                print(f'----------    iteration    {iteration}    ----------')
                
            try:
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)
            except StopIteration:
                train_iter = iter(train_dl)
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)

            model.train()
            ema.train()

            optim.zero_grad()

            loss = self.loss(model=model, x0=x0)
            print(f'Loss:  {loss.item()}')
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            print(f'Grad norm: {grad_norm.item()}')

            optim.step()
            ema.update_parameters(model)


            if iteration % 5000 == 0:
                ema.eval()
                loss = 0

                with torch.no_grad():
                    for x0, _ in test_dl:
                        x0 = x0.to(self.device)

                        loss += self.loss(model=ema, x0=x0).detach()

                avg_loss = loss.item() / len(test_dl)
                print(f'>>>>>>>>>> avg test loss:    {avg_loss}')


            # regularly sample a small grid to check progress
            if (iteration + 1) % 10_000 == 0:
                ema.eval()
    
                samples = self.sample(model=ema, amount=64)    # are [-1, 1]
                samples = (samples + 1.0) * 0.5    # now [0, 1]
                samples = samples.clamp(0.0, 1.0)    # for good measure

                grid = torchvision.utils.make_grid(samples, nrow=8, padding=2, normalize=False)
                plt.figure(figsize=(8, 8))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                plt.axis("off")
        
                plt.savefig(os.path.join(self.grid_path, f'{iteration}.png'), dpi=200, bbox_inches="tight", pad_inches=0)
                
                plt.close()
    
                print(f'-----------------------------------------------generated an 8x8 grid and saved it to:  {self.grid_path}')

            if (iteration + 1) % 50_000 == 0:

                ema.eval()
                samples = self.sample(model=ema, amount=2_000)

                gen_ds = Uint8Dataset(to_uint8_rgb(samples, self.data))

                metrics = torch_fidelity.calculate_metrics(
                    input1=eval_dl,
                    input2=gen_ds,
                    batch_size=256,
                    fid=True,
                    cuda=(('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda'),
                    verbose=False,
                )
                ema_score = metrics['frechet_inception_distance']

                print(f"Tested the ema model. FID Score (2_000 samples): {ema_score:.4f}")

                if ema_score < self.best_score:
                    self.best_score = ema_score

                    score_save_path = os.path.join(self.base, 'best_score_model.pth')

                    uncompiled_model = getattr(ema.module, "_orig_mod", ema.module)
                    torch.save(uncompiled_model.state_dict(), score_save_path)
                    print(f"saved best score model to:  {score_save_path},    score {ema_score}")

                    
        # ! Final Fid evaluation on 50_000 samples

        eval_ds = self.data.get_dataset_for_full_eval()

        ema.eval()
        samples = self.sample(model=ema, amount=50_000)

        gen_ds = Uint8Dataset(to_uint8_rgb(samples, self.data))

        metrics = torch_fidelity.calculate_metrics(
            input1=eval_ds,
            input2=gen_ds,
            batch_size=256,
            fid=True,
            cuda=(('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda'),
            verbose=False,
        )
        ema_score = metrics['frechet_inception_distance']

        print(f"Tested the ema model. FID Score (50_000 samples): {ema_score:.4f}")

if __name__ == '__main__':

    DDPM_pp_cont = Diffusion()
    DDPM_pp_cont.train()