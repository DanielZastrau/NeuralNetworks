import os
import argparse

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.velo_utils import compute_velocity
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class Kac():
    """2026 - Duong & Chemseddine - Telegraphers Generative Model via Kac Flows
    No point in reimplementing the base model of 2026 - Han et al - DistillKac,
            as it is only a deeper unet.
    
    Later I want to see if I can add some of the diffusion modulations.

    Their best 50k FID with S=100:    6.42

    Our model + Euler stepping + uniform time steps
        Best 50k FID with S=100:    4.8
        Best 2k FID with S=100:    28.4

    Our model + Euler stepping + Karras schedule
        Best 50k FID with S=100:    299.3 ???

    Our model + Karras sampler + uniform steps
        Best 50k FID with S=50:

    - Karras schedule    -    Base model 1 just different sampling
    - Karras Heun    -    Base model 1 just different sampling
    - Data augmentation    -    Base model 2
    - Han et als model settings    -    Base model 3
    """

    def __init__(self, time_steps: str = 'uniform', integrator: str = 'euler'):

        assert time_steps in ['uniform', 'karras']
        assert integrator in ['euler', 'karras']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,)
        )

        self.a = 25
        self.c = 2

        self.sampler = TorchKacConstantSampler(
            a=self.a,
            c=self.c,
            T=1,
            M=50_000,
            K=4_096,
        )

        self.iterations = 400_000

        self.lr = 2e-4
        self.lr_warmup = int(self.iterations * 0.05)

        self.epsilon = 1e-5    # time truncation
        self.T = 1    # max time

        self.time_steps = time_steps
        self.integrator = integrator
        self.S = 50    # amount of sampling steps

        self.karras_p = 7    # staying with the choice of 2022 - Karras - Elucidating the design space of diffusion models

        self.base = '/work/zastrau/2026Duong'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.curr_dir = os.path.join(self.base, 'simple')
        if not os.path.exists(self.curr_dir):
            os.mkdir(self.curr_dir)

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

    def f(self, t: torch.Tensor):

        return 1 - t

    def df(self, t: torch.Tensor):

        return torch.ones_like(t, device=self.device) * -1

    def g(self, t: torch.Tensor):

        return t

    def dg(self, t: torch.Tensor):

        return torch.ones_like(t, device=self.device)

    def get_karras_schedule(self) -> list[float]:

        t_values = [
            (self.T**(1/self.karras_p) + (i / (self.S - 1)) * (self.epsilon**(1/self.karras_p) - self.T**(1/self.karras_p)))**self.karras_p 
            for i in (range(self.S))
        ] + [0.0]

        return t_values

    def get_model(self):

        #! The corrected network config of Duong et al
        self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=128, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=2, dropout=0.1,
                         attention_resolutions=(2,), num_heads=4, use_new_attention_order=True, ).to(self.device)

    def get_ema(self):

        self.ema = torch.optim.swa_utils.AveragedModel(self.model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999)).to(self.device)

    def get_optim(self):

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optim,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer=self.optim,
                                                  start_factor=0.2,
                                                  end_factor=1.0,
                                                  total_iters=self.lr_warmup),
                torch.optim.lr_scheduler.ConstantLR(optimizer=self.optim,
                                                    factor=1.0,
                                                    total_iters=1),
            ],
            milestones=[self.lr_warmup]
        )

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        # [B,]
        t = torch.rand((x0.shape[0],), device=self.device)

        # [B,]
        ft = self.f(t=t)
        dft = self.df(t=t)
        gt = self.g(t=t)
        dgt = self.dg(t=t)

        # [B, C x H x W]
        z = self.sampler.sample(gt, dim=self.data.data_dims.total_dimension).to(self.device)

        # [B, C, H, W]
        z = z.reshape(x0.shape)

        # [B, C, H, W] = [B,] * [B, C, H, W] + [B, C, H, W]
        xt = ft.view(-1, *([1] * (x0.dim() - 1))) * x0 + z

        # [B, C, H, W] = [B,] * [B, C, H, W]
        drift = dft.view(-1, *([1] * (x0.dim() - 1))) * x0
        with torch.no_grad():
            # [B, C, H, W],    retains the shape of z,    shape of x must match shape of t
            velo = dgt.view(-1, *([1] * (x0.dim() - 1))) * compute_velocity(
                x=z,
                t=gt.view(-1, *([1] * (x0.dim() - 1))),
                a=self.a,
                c=torch.tensor(self.c),
                epsilon=self.epsilon
            )

        # [B, C, H, W]
        pred = model(xt, t * 1000)

        target = velo + drift

        return torch.nn.functional.mse_loss(pred, target)

    def sample(self, model: torch.nn.Module, amount: int):

        samples = []

        with torch.no_grad():
            for j in range((amount // 512) + 1):
                how_many = min(512, amount - j * 512)

                # [B, C x H x W]
                xT = self.sampler.sample(t = torch.ones((how_many,), device=self.device, dtype=torch.float32), dim=self.data.data_dims.total_dimension).to(self.device)

                # [B, C, H, W]
                xT = xT.reshape((
                    how_many,
                    self.data.data_dims.channels,
                    self.data.data_dims.height,
                    self.data.data_dims.width
                ))
                xt = xT

                dt = (1 - self.epsilon) / self.S

                if self.time_steps == 'uniform':
                    time_steps = torch.linspace(1, self.epsilon + dt, self.S, device=self.device, dtype=torch.float32)
                else:    # self.time_steps == 'karras':
                    time_steps = self.get_karras_schedule()

                for i in range(len(time_steps) - 1):

                    if self.integrator == 'euler':

                        t = time_steps[i]

                        t_tensor = torch.full((xT.shape[0],), float(t), device=self.device, dtype=torch.float32)
                        pred_v = model(xt, t_tensor * 1000)
                        xt = xt - pred_v * dt

                    elif self.integrator == 'karras':

                        ti = time_steps[i]
                        tip = time_steps[i + 1]
                        diff = tip - ti

                        # ? Evaluate velocity at ti (which is equivalent to evaluating the pfode at ti)
                        pred_v_i = model(xt, ti * 1000)

                        # ? Euler step
                        x_intermediate = xt + diff * pred_v_i

                        # ? Second order correction
                        if tip != 0:
                            pred_v_ip = model(x_intermediate, tip * 1000)
                            xt = xt + diff * (0.5 * pred_v_i + 0.5 * pred_v_ip)

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
        for iteration in range(self.iterations):
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
    args = parser.parse_args()

    model = Kac()
    if args.what == 'full' or args.what == 'train':
        model.train()

    if args.what == 'full' or args.what == 'eval':
        model.eval()