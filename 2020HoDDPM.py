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
    """2020 - Ho et al - Denoising Diffusion Probabilistic Models
    Their best Cifar-10 50k Fid was 3.17
    
    Best 2k Fid with T=1_000 was 26.7
    """

    def __init__(self, sigma_choice: str = 'simple'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.T = 1000
        self.betas = torch.linspace(0.0001, 0.02, self.T)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sigmas = self.get_sigmas(choice = sigma_choice)

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000, training_evaluation_period_fid_num_samples = 2_000))


        self.base = '/work/zastrau/diffusionHo'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.curr_dir = os.path.join(self.base, sigma_choice)
        if not os.path.exists(self.curr_dir):
            os.mkdir(self.curr_dir)

        self.grid_path = os.path.join(self.curr_dir, 'grids')

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.curr_dir, 'best_score_model.pth')

    def get_sigmas(self, choice: str = 'simple'):

        if choice == 'simple':
            return torch.sqrt(self.betas)

        else:    # choice == 'other':

            # Shift alphas_bar by 1 to represent alphas_bar_{t-1}, setting alpha_bar_0 = 1.0
            alphas_bar_prev = torch.cat([torch.tensor([1.0]), self.alphas_bar[:-1]])

            return ((1.0 - alphas_bar_prev) / (1.0 - self.alphas_bar)) * self.betas

    def get_model(self):

        #! the network natively implements group normalization
        return UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=128, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=2, attention_resolutions=(2,),
                         dropout=0.1,).to(self.device)

    def get_ema(self, model:torch.nn.Module):

        return torch.optim.swa_utils.AveragedModel(model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999)).to(self.device)

    def get_optim(self, model: torch.nn.Module):

        return torch.optim.Adam(model.parameters(), lr=2e-4)

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        z = torch.randn_like(x0)

        alpha_bar_t = self.alphas_bar[t]
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (x0.dim() - 1)))

        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * z

        pred = model(xt, t)

        return torch.nn.functional.mse_loss(pred, z)

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

            xt = xT
            for t in range(self.T-1, -1, -1):

                if t > 0:
                    z = torch.randn_like(xT)
                else:
                    z = torch.zeros_like(xT)

                alpha_t = self.alphas[t]
                alpha_bar_t = self.alphas_bar[t]
                sigma_t = self.sigmas[t] 

                prefactor = torch.sqrt(alpha_t) ** (-1)
                postsummand = sigma_t * z
                numerator = (1 - alpha_t)
                denominator = torch.sqrt(1 - alpha_bar_t)

                t_tensor = torch.full((xt.shape[0],), t, dtype=torch.long, device=xt.device)
                
                with torch.no_grad():
                    pred = model(xt, t_tensor)

                xt = prefactor * (xt - (numerator / denominator) * pred) + postsummand   

            if amount == 50_000:
                print(f'sampled {i * 512 + how_many} / 50_000')
            samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    def train(self):

        model = self.get_model()
        ema = self.get_ema(model=model)
        optim = self.get_optim(model=model)

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for iteration in range(800_000):
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
    parser.add_argument('--sigmas', type=str, choices=['simple', 'other'], default='simple')
    parser.add_argument('--what', type=str, choices=['full', 'train', 'eval'], default='full')
    args = parser.parse_args()

    DDPM = Diffusion(sigma_choice=args.sigmas)
    if args.what == 'full' or args.what == 'train':
        DDPM.train()

    if args.what == 'full' or args.what == 'eval':
        DDPM.eval()