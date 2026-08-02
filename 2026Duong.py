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
from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.velo_utils import compute_velocity
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class Kac():
    """2026 - Duong & Chemseddine - Telegraphers Generative Model via Kac Flows
    No point in reimplementing the base model of 2026 - Han et al - DistillKac,
            as it is only a deeper unet.
    
    Later I want to see if I can add some of the diffusion modulations.

    Their best 50k FID with S=100:    6.42

    Base model 1 + Euler integrator (S=50) + uniform schedule
        50k FID with S=100:    4.8
        Best 2k FID with S=100:    28.4

    Base model 1 + Euler integrator (S=50) + Karras schedule
        50k FID with S=100:    7.1732

    Base model 1 + Karras integrator (S=50) + karras schedule
        Best 50k FID with S=50:    10.8887

    Thoughs:
        - Karras schedule and integrator seem to yield worse results
                That could be due to this script implementing a VP Kac process, while Karras implemented a VE Diffusion process.

    - Data augmentation    -    Base model 2
    - Han et als model settings    -    Base model 3
    """

    def __init__(self, which: str = 'simple', schedule: str = 'uniform', integrator: str = 'euler'):

        assert schedule in ['uniform', 'karras']
        assert integrator in ['euler', 'heun']

        self.which = which

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

        self.augmentation_pipeline = KarrasAugmentationPipeline()

        self.I = 400_000

        self.lr = 2e-4
        self.lr_warmup = int(self.I * 0.05)

        self.epsilon = 1e-5    # time truncation
        self.T = 1    # max time

        self.schedule = schedule
        self.integrator = integrator

        self.model_channels = 128

        if self.integrator == 'euler':
            self.S = 100    # amount of sampling steps
        else:    # self.integrator == 'heun'
            self.S = 50

        self.karras_p = 7    # staying with the choice of 2022 - Karras - Elucidating the design space of diffusion models

        self.base = '/work/zastrau/2026Duong'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.curr_dir = os.path.join(self.base, which)
        if not os.path.exists(self.curr_dir):
            os.mkdir(self.curr_dir)

        self.grid_path = os.path.join(self.curr_dir, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.curr_dir, 'best_score_model.pth')

    def f(self, t: torch.Tensor):

        return 1 - t

    def df(self, t: torch.Tensor):

        return torch.ones_like(t, device=self.device) * -1

    def g(self, t: torch.Tensor):

        return t

    def dg(self, t: torch.Tensor):

        return torch.ones_like(t, device=self.device)

    def get_karras_schedule(self, N: int) -> list[float]:

        t_values = [
            (self.T**(1/self.karras_p) + (i / (N - 1)) * (self.epsilon**(1/self.karras_p) - self.T**(1/self.karras_p)))**self.karras_p 
            for i in (range(N))
        ] + [0.0]

        return t_values

    def model_fn(self, model: torch.nn.Module, t: torch.Tensor | float, x:torch.Tensor, aug_cond: torch.Tensor | None):

        if not isinstance(t, torch.Tensor) or t.dim() == 0:
            t = torch.full((x.shape[0],), float(t), dtype=torch.float32, device=self.device)
        elif t.shape[0] != x.shape[0]:
            t = t.expand((x.shape[0], ))

        active_model = getattr(model, "module", model)

        t_emb = timestep_embedding(t * 1_000, self.model_channels)
        emb = active_model.time_embed(t_emb)

        if aug_cond is None:
            aug_cond = torch.zeros((x.shape[0], 9), dtype=torch.float32, device=self.device)

        emb = emb + active_model.aug_proj(aug_cond)

        pred = model(x, timesteps = None, emb_override=emb)

        return pred

    def get_model(self):

        #! The corrected network config of Duong et al
        self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=self.model_channels, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=2, dropout=0.1,
                         attention_resolutions=(2,), num_heads=4, use_new_attention_order=True,).to(self.device)

        if self.which == 'model2':
            self.model.aug_proj = torch.nn.Linear(9, self.model_channels * 4, device=self.device).to(self.device)

    def get_ema(self):

        self.ema = torch.optim.swa_utils.AveragedModel(self.model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999)).to(self.device)

    def get_optim(self):

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optim,
                                                  start_factor=0.2,
                                                  end_factor=1.0,
                                                  total_iters=self.lr_warmup)

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        if self.which == 'simple':
            x0_aug, aug_cond = x0, None
        else:    # self.which == 'model2':
            x0_aug, aug_cond = self.augmentation_pipeline(x0)

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
        xt = ft.view(-1, *([1] * (x0.dim() - 1))) * x0_aug + z

        # [B, C, H, W] = [B,] * [B, C, H, W]
        drift = dft.view(-1, *([1] * (x0.dim() - 1))) * x0_aug
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
        pred = self.model_fn(model=model, t=t, x=xt, aug_cond=aug_cond)

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

                if self.integrator == 'euler':

                    if self.schedule == 'uniform':
                        dt = (1 - self.epsilon) / self.S
                        time_steps = torch.linspace(1, self.epsilon + dt, self.S, device=self.device, dtype=torch.float32)

                    else:    # self.time_steps == 'karras':
                        #! Cut off the trailing two values — which are roughly eps and 0 — so that the Euler integrator takes
                        #!      one final ever so little step to zero
                        #! Also add a step to the schedule, so that the S-th step is one before eps. As it would be in the
                        #!      uniform schedule
                        time_steps = torch.tensor(self.get_karras_schedule(self.S + 1)[:-2], device=self.device, dtype=torch.float32)

                    for i in range(len(time_steps)):

                        t = time_steps[i]

                        if i < len(time_steps) - 1:
                            dt = t - time_steps[i + 1]
                        else:
                            dt = t - self.epsilon

                        pred_v = self.model_fn(model=model, t=t, x=xt, aug_cond=None)
                        xt = xt - pred_v * dt

                elif self.integrator == 'heun':

                    if self.schedule == 'uniform':
                        dt = (1 - self.epsilon) / self.S
                        time_steps = torch.linspace(1, self.epsilon, self.S + 1, device=self.device, dtype=torch.float32)

                    else:    # self.schedule == 'karras'
                        time_steps = torch.tensor(self.get_karras_schedule(self.S))

                    for i in range(len(time_steps) - 1):

                        ti = time_steps[i]
                        tip = time_steps[i + 1]

                        dt = tip - ti

                        # ? Evaluate velocity at ti and take a euler step
                        pred_v_i = self.model_fn(model=model, t=ti, x=xt, aug_cond=None)
                        x_intermediate = xt + dt * pred_v_i

                        # ? Second order correction
                        if tip != 0:
                            pred_v_ip = self.model_fn(model=model, t=tip, x=x_intermediate, aug_cond=None)
                            xt = xt + dt * (0.5 * pred_v_i + 0.5 * pred_v_ip)

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
    parser.add_argument('--which', type=str, default='simple', choices=['simple', 'model2'])
    parser.add_argument('--schedule', type=str, default='uniform', choices=['uniform', 'karras'])
    parser.add_argument('--integrator', type=str, default='euler', choices=['euler', 'heun'])
    args = parser.parse_args()

    model = Kac(which=args.which, schedule=args.schedule, integrator=args.integrator)
    if args.what == 'full' or args.what == 'train':
        model.train()

    if args.what == 'full' or args.what == 'eval':
        model.eval()