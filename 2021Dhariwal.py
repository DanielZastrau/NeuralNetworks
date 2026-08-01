import os
import argparse

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.networks.neuralNetworkOpenAI  import UNetModel

class Diffusion():
    """2021 - Nichol & Dhariwal - Improved DDPM

    Improvements / Differences include:
        - cosine schedule
        - T = 4_000 instead of 1_000
        - Hybrid training objective
        - also learning the variance
        - larger model

    Their reported 50k FID score:  ~3.19  
    Our achieved 50k FID score:  3.9
    Our minimum 2k FID score:  27.4
    """

    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.T = 4_000
        self.S = 4_000

        t = torch.linspace(0, 1, self.T, device=self.device)
        s = 0.008 
        self.cosine_values = (torch.cos(0.5 * torch.pi * (t + s) / (1 + s))**2).to(self.device)

        # Directly specifying the alphas_bar
        self.alphas_bar = (self.cosine_values / self.cosine_values[0]).to(self.device)
        self.alphas_bar_previous = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_bar[:-1]]).to(self.device)

        # Defining the noise schedule in terms of alphas_bar
        self.betas = (1 - (self.alphas_bar / self.alphas_bar_previous)).to(self.device)
        self.betas = self.betas.clamp(min=1e-5, max=0.999).to(self.device)

        # the usual identity alphas = 1 - betas
        self.alphas = (1.0 - self.betas).to(self.device)

        # Recalculating the alpha_bar values to avoid numerical error accumulation
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_bar_previous = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_bar[:-1]]).to(self.device)

        # posterior is q(x_{t-1} \mid x_t, x_0) = N(x_{t-1} \mid \tilde \mu_t(x_t, x_0), \tilde \beta_t I_d)
        self.betas_tilde = (self.betas * (1.0 - self.alphas_bar_previous) / (1.0 - self.alphas_bar)).to(self.device)
        self.mus_tilde_coefficient_one = (self.betas * torch.sqrt(self.alphas_bar_previous) / (1.0 - self.alphas_bar)).to(self.device)
        self.mus_tilde_coefficient_two = ((1.0 - self.alphas_bar_previous) * torch.sqrt(self.alphas) / (1.0 - self.alphas_bar)).to(self.device)

        # Needed for Variance calculation \Sigma_\theta(t, xt) = \exp(v \log \beta_t + (1 - v) \log \tilde \beta_t).
        self.log_betas = torch.log(self.betas).to(self.device)
        self.log_betas_tilde = torch.log(self.betas_tilde.clamp(min=1e-20)).to(self.device)

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,
        ))

        self.base = '/work/zastrau/Dhariwal'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

    def get_model(self):

        return UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels * 2,
                         model_channels=128, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=3, dropout=0.3,
                         attention_resolutions=(2, 4), num_heads=4,
                         use_scale_shift_norm=True).to(self.device)

    def get_ema(self, model: torch.nn.Module):

        return torch.optim.swa_utils.AveragedModel(model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999)).to(self.device)

    def get_optim(self, model: torch.nn.Module):

        return torch.optim.Adam(model.parameters(), lr=1e-4)

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        noise = torch.randn_like(x0, device=self.device)

        alpha_t = self.alphas[t].view(-1, *([1] * (x0.dim() - 1)))
        alpha_bar_t = self.alphas_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        beta_t = self.betas[t].view(-1, *([1] * (x0.dim() - 1)))

        mu_tilde_coeff_one_t = self.mus_tilde_coefficient_one[t].view(-1, *([1] * (x0.dim() - 1)))
        mu_tilde_coeff_two_t = self.mus_tilde_coefficient_two[t].view(-1, *([1] * (x0.dim() - 1)))

        log_beta_tilde_t = self.log_betas_tilde[t].view(-1, *([1] * (x0.dim() - 1)))
        log_beta_t = self.log_betas[t].view(-1, *([1] * (x0.dim() - 1)))

        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

        # ! t needs to be between 0 and 1k 
        out = model(x_t, t.float())
        pred_noise, pred_v = out[:, :self.data.data_dims.channels], out[:, self.data.data_dims.channels:]
        
        loss_simple = torch.nn.functional.mse_loss(pred_noise, noise, reduction='none').mean(dim=1)

        # We have access to x0 and xt so we can calculate the true mean at t-1
        # \tilde \mu_t(x_t, x_0) = \frac{\sqrt{\bar \alpha_{t-1}}\beta_t}{1 - \bar \alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar \alpha_{t-1})}{1 - \bar \alpha_t}x_t
        true_mean = mu_tilde_coeff_one_t * x0 + mu_tilde_coeff_two_t * x_t

        # The predicted mean is calculated via the proxy pred_noise
        # \mu_\theta(t, x_t) = (\sqrt{\alpha_t})^{-1} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar \alpha_t}} \varepsilon_\theta(t, x_t) \right).
        pred_mean = torch.sqrt(alpha_t)**(-1) * (x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise)
        pred_mean_sg = pred_mean.detach()

        # The predicted variance is calculated via the proxy pred_v. Calculate in two steps, because the inner term is needed for the KL calculation
        # \Sigma_\theta(t, xt) = \exp(v \log \beta_t + (1 - v) \log \tilde \beta_t).
        pred_log_variance = pred_v * log_beta_t + (1.0 - pred_v) * log_beta_tilde_t
        pred_variance = torch.exp(pred_log_variance)
        
        kl = 0.5 * (pred_log_variance - log_beta_tilde_t + (torch.exp(log_beta_tilde_t) + (true_mean - pred_mean_sg)**2) / pred_variance - 1.0)
        
        # Mask out t=0 to prevent singular KL explosion
        loss_vlb = torch.where((t == 0).view(-1, *([1] * (x0.dim() - 1))), torch.zeros_like(kl, device=self.device), kl).mean(dim=1)
        
        lambda_weight = 0.001
        return (loss_simple + lambda_weight * loss_vlb).mean()

    def sample(self, model: torch.nn.Module, amount: int):

        samples = []

        for j in range((amount // 512) + 1):
            how_many = min(512, amount - j * 512)

            xT = torch.randn((how_many,
                              self.data.data_dims.channels,
                              self.data.data_dims.height,
                              self.data.data_dims.width),
                              device=self.device,
                              dtype=torch.float32)
            xt = xT

            steps = torch.round(torch.linspace(self.T - 1, 0, self.S, device=self.device)).long()

            with torch.no_grad():
                for i, t in enumerate(steps):
                    
                    t_tensor = torch.full((xt.shape[0],), t.item(), dtype=torch.long, device=self.device)

                    # Determine the previous step in the subsequence (S_{t-1})
                    t_prev = steps[i + 1] if i < len(steps) - 1 else torch.tensor(-1, device=self.device)

                    z = torch.randn_like(xt, device=self.device) if t > 0 else torch.zeros_like(xt, device=self.device)
                    
                    out = model(xt, t_tensor.float())
                    pred_noise, pred_v = out[:, :self.data.data_dims.channels], out[:, self.data.data_dims.channels:]
                    
                    # Fetch \bar{\alpha} for current and previous subsequence steps
                    alpha_bar_t = self.alphas_bar[t]
                    alpha_bar_t_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
                    
                    # Recompute schedule parameters for the skipped interval
                    beta_S_t = (1.0 - (alpha_bar_t / alpha_bar_t_prev)).to(self.device)
                    alpha_S_t = (1.0 - beta_S_t).to(self.device)
                    beta_tilde_S_t = (beta_S_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)).to(self.device)
                    
                    # Calculate log variances for the subsequence
                    log_beta_S_t = torch.log(beta_S_t).to(self.device)
                    log_beta_tilde_S_t = torch.log(beta_tilde_S_t.clamp(min=1e-20)).to(self.device)

                    # The predicted mean uses the recomputed alpha_S_t and beta_S_t
                    pred_mean = torch.sqrt(alpha_S_t)**(-1) * (xt - beta_S_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise)

                    # The predicted standard deviation uses the recomputed log variances
                    pred_sigma = torch.exp(0.5 * (pred_v * log_beta_S_t + (1.0 - pred_v) * log_beta_tilde_S_t))

                    # Ancestral sampling step
                    xt = pred_mean + pred_sigma * z

            if amount == 50_000:
                print(f'Sampled {j * 512 + how_many} / 50_000.')
            samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    def train(self):
    
        model = self.get_model()
        ema = self.get_ema(model=model)
        optim = self.get_optim(model=model)

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for iteration in range(500_000):
            if iteration % 1_000 == 0:
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

            grad_norm = torch.sqrt(sum(p.grad.data.norm() ** 2 for p in model.parameters() if p.grad is not None))
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
                print(f'>>>>>>>>>> Avg test loss:    {avg_loss}.')


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

                    uncompiled_model = getattr(ema.module, "_orig_mod", ema.module)
                    torch.save(uncompiled_model.state_dict(), self.score_save_path)
                    print(f"saved best score model to:  {self.score_save_path},    score {ema_score}")

    def eval(self):
                    
        # ! Final Fid evaluation on 50_000 samples

        # Load the best model
        model = self.get_model()
        model.load_state_dict(torch.load(self.score_save_path, map_location=self.device))

        eval_ds = self.data.get_dataset_for_full_eval()

        model.eval()
        samples = self.sample(model=model, amount=50_000)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--what', type=str, choices=['full', 'train', 'eval'], default='full')
    args = parser.parse_args()

    iDDPM = Diffusion()
    if args.what == 'full' or args.what == 'train':
        iDDPM.train()

    if args.what == 'full' or args.what == 'eval':
        iDDPM.eval()