import os
import argparse

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.utils.dataAugmentation import KarrasAugmentationPipeline
from Cluster.utils.nn_utils import timestep_embedding
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class EDM():
    """2022 - Karras et al - Elucidating the design space of diffusion-based generative models

    Specific to their work:
        - This implements the Karras schedule.
        - This implements the parameterization of the network in terms of the functions "c...".
        - This implements the geometric transformations.
        - The zero output layer is natively implemented by the UNet
        - This implements the Heun pfODE solver

    Our implementation:
        Min 2k fid with S=50:    26.1
        50k fid with S=50:    2.65

    Their best 50k fid with S=18:    ~1.98

    
    Differences to their work:
        - I am not training with a batch size of 512, because I have found that to cause an OOM error.
                I am training with a batch size of 128 instead to stay consistenc with everything else.
        - Consequently their 400k iterations would become 1.6m iterations.
                I cap them at 1m iterations, because I have found no benefits in the later stages of training.
        - Because we train with a smaller batch size, we also have to adjust their learning rate by the same factor of 4.
                Down from 1e-3 to 2.5e-4

    """

    def __init__(self, S: int = 50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,)
        )

        self.augmentation_pipeline = KarrasAugmentationPipeline()

        self.sigma_min = 0.002
        self.sigma_max = 80
        self.sigma_data = 0.5
        
        self.karras_p = 7.0

        self.P_mean = -1.2
        self.P_std = 1.2

        self.I = 1_000_000    # number of training iterations
        self.S = S    # number of sampling steps

        self.time_factor = 1_000 / self.sigma_max

        self.lr = 2.5e-4    # originally 1e-3 but since we train with a quarter of the batch size, the lr has to adjust as well
        self.lr_warmup = int(self.I * 0.05)

        self.model_channels = 128

        self.base = '/work/zastrau/2022Karras'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

    def sigma(self, t: float):

        return t

    def c_skip(self, sigma: torch.Tensor):

        return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2)).to(self.device)

    def c_out(self, sigma: torch.Tensor):

        return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma **2)).to(self.device)

    def c_in(self, sigma: torch.Tensor):

        return (1 / (torch.sqrt(self.sigma_data**2 + sigma**2))).to(self.device)

    def c_noise(self, sigma: torch.Tensor):

        return (0.25 * torch.log(sigma)).to(self.device)

    def weight(self, sigma: torch.Tensor):

        return ((sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2).to(self.device)

    def D_fn(self, model: torch.nn.Module, x: torch.Tensor, sigma: torch.Tensor | float, aug_cond: torch.Tensor | None):

        if isinstance(sigma, float):
            sigma = torch.full((x.shape[0],), sigma, dtype=torch.float32, device=self.device)

        sigma_bc = sigma.view(-1, *([1] * (x.dim() - 1)))
        
        cin = self.c_in(sigma=sigma_bc)
        cskip = self.c_skip(sigma=sigma_bc)
        cout = self.c_out(sigma=sigma_bc)
        cnoise = self.c_noise(sigma=sigma).flatten()

        active_model = getattr(model, "module", model)

        t_emb = timestep_embedding(cnoise * self.time_factor, self.model_channels)
        emb = active_model.time_embed(t_emb)

        if aug_cond is None:
            aug_cond = torch.zeros((x.shape[0], 9), dtype=torch.float32, device=self.device)

        emb = emb + active_model.aug_proj(aug_cond)

        pred = model(cin * x, timesteps = None, emb_override=emb)

        return cskip * x + cout * pred

    def get_karras_schedule(self) -> list[float]:

        t_values = [
            (self.sigma_max**(1/self.karras_p) + (i / (self.S - 1)) * (self.sigma_min**(1/self.karras_p) - self.sigma_max**(1/self.karras_p)))**self.karras_p 
            for i in (range(self.S))
        ] + [0.0]

        return t_values

    def get_model(self):

        #! the network natively implements group normalization
        self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=128, channel_mult=(2, 2, 2),
                         num_res_blocks=4, attention_resolutions=(2,),
                         dropout=0.13, use_new_attention_order=True,)

        self.model.aug_proj = torch.nn.Linear(9, model.model_channels * 4)

        self.model.to(self.device)

    def get_ema(self):

        self.ema = torch.optim.swa_utils.AveragedModel(self.model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9998)).to(self.device)

    def get_optim(self):

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optim,
                                            start_factor=0.2,
                                            end_factor=1.0,
                                            total_iters=self.lr_warmup)

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        x0_aug, aug_cond = self.augmentation_pipeline(x0)

        sigma = torch.exp(torch.randn((x0.shape[0],), device=self.device) * self.P_std + self.P_mean)
        sigma_bc = sigma.view(-1, *([1] * (x0.dim() - 1)))

        z = torch.randn_like(x0, device=self.device) * sigma_bc
        xt = x0_aug + z

        pred = self.D_fn(model=model, x=xt, sigma=sigma, aug_cond=aug_cond)
        weight = self.weight(sigma_bc)

        return (weight * torch.nn.functional.mse_loss(pred, x0_aug, reduction='none')).mean()

    def sample(self, model: torch.nn.Module, amount: int):

        samples = []

        with torch.no_grad():
            for j in range((amount // 512) + 1):
                how_many = min(512, amount - j * 512)

                t_values: list[float] = self.get_karras_schedule()

                xT = torch.randn((how_many,
                                self.data.data_dims.channels,
                                self.data.data_dims.height,
                                self.data.data_dims.width),
                                device=self.device,
                                dtype=torch.float32) * self.sigma_max

                xt = xT
                for i in range(len(t_values) - 1):

                    ti = t_values[i]
                    tip = t_values[i + 1]
                    sigma_ti = self.sigma(ti)
                    sigma_tip = self.sigma(tip)
                    diff = tip - ti

                    # ? pfode evaluation at the current timestep
                    dt = ( 1 / sigma_ti ) * xt - (1 / sigma_ti) * self.D_fn(model=model, x=xt, sigma=sigma_ti, aug_cond=None)

                    # ? Euler step
                    x_intermediate = xt + diff * dt

                    # ? Second order correction
                    if sigma_tip != 0:
                        dtprime = ( 1 / sigma_tip ) * x_intermediate - (1 / sigma_tip) * self.D_fn(model=model, x=x_intermediate, sigma=sigma_tip, aug_cond=None)

                        xt = xt + diff * (0.5 * dt + 0.5 * dtprime)

                    else:
                        xt = x_intermediate

                if amount == 50_000:
                    print(f'sampled {j * 512 + how_many} / 50_000')
                samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    def train(self):

        self.get_model()
        self.get_ema()
        self.get_optim()

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for iteration in range(self.I):
            if iteration % 1_000 == 0:
                print(f'----------    iteration    {iteration}    ----------')

            try:
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)
            except StopIteration:
                train_iter = iter(train_dl)
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)

            self.model.train()
            self.ema.train()

            self.optim.zero_grad()

            loss = self.loss(model=self.model, x0=x0)
            print(f'Loss:  {loss.item()}')
            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.data.norm() ** 2 for p in self.model.parameters() if p.grad is not None))
            print(f'Grad norm: {grad_norm.item()}')

            self.optim.step()
            self.scheduler.step()
            self.ema.update_parameters(self.model)

            if (iteration + 1) % 5000 == 0:
                self.ema.eval()
                loss = 0

                with torch.no_grad():
                    for x0, _ in test_dl:
                        x0 = x0.to(self.device, dtype=torch.float32)

                        loss += self.loss(model=self.ema, x0=x0).detach()

                avg_loss = loss.item() / len(test_dl)
                print(f'>>>>>>>>>> avg test loss:    {avg_loss}')


            # regularly sample a small grid to check progress
            if (iteration + 1) % 10_000 == 0:
                self.ema.eval()
    
                samples = self.sample(model=self.ema, amount=64)    # are [-1, 1]
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
                self.ema.eval()

                samples = self.sample(model=self.ema, amount=2_000)

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

                    uncompiled_model = getattr(self.ema.module, "_orig_mod", self.ema.module)
                    torch.save(uncompiled_model.state_dict(), self.score_save_path)
                    print(f"saved best score model to:  {self.score_save_path},    score {ema_score}")

    def eval(self):
                    
        # ! Final Fid evaluation on 50_000 samples

        # Load the best model
        self.get_model()
        self.model.load_state_dict(torch.load(self.score_save_path, map_location=self.device))

        eval_ds = self.data.get_dataset_for_full_eval()

        self.model.eval()
        samples = self.sample(model=self.model, amount=50_000)

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
    parser.add_argument('--S', type=int, default=50)
    args = parser.parse_args()

    model = EDM(S=S)
    if args.what == 'full' or args.what == 'train':
        model.train()

    if args.what == 'full' or args.what == 'eval':
        model.eval()