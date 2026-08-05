import os
import argparse

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.dataAugmentation import KarrasAugmentationPipeline
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.networks.neuralNetworkOpenAI import UNetModel
from Cluster.utils.nn_utils import timestep_embedding

class DSBFM():
    """DSB-FM as describe in my master thesis

    [01.08.26]    Run: zastrau     Reduced LR to 1e-4
            Min 2k FID at 400k iterations:    ~54
            With the reduced learning rate the Fid score trajectory also looks better, and is more stably trending downward
    [01.08.26]    Run: zastrau2    Added weighting
            Min 2k FID at 400k iterations:    ~55
    [02.08.26]    Run: zastrau3    Added gradient clipping
            Min 2k FID at 400k iterations:    ~34 @ 150k iterations  >  after that followed degradation
            50k FID:    10.9
    [02.08.26]    Run: zastrau4    Added Karras augmentation to fight the degradation / overfitting
            Min 2k FID at 400k iterations:    ~33 ~ 400k iterationos
            50k FID:    9.2


    In the future I want to apply some of the diffusion improvements here too.
    But my expectation is that it yields close to the same results, since it
    basically is a diffusion model.

    I think the improvements due to Karras 2022 should be largly applicable
    """

    def __init__(self, which: str = 'simple'):

        self.which = which

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,)
        )

        self.augmentation_pipeline = KarrasAugmentationPipeline()

        self.I = 400_000    # amount of training iterations
        self.S = 1_024    # amount of sampling steps
        self.lr = 2e-4
        self.lr_warmup = int(self.I * 0.05)
        self.epsilon = 1e-5

        self.model_channels = 128

        self.base = '/work/zastrau/2026Zastrau'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.curr_dir = os.path.join(self.base, which)
        if not os.path.exists(self.curr_dir):
            os.mkdir(self.curr_dir)

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

    def f(self, t: torch.Tensor):

        return 1 - t

    def g(self, t: torch.Tensor):

        return torch.sqrt(t)

    def weight(self, t: torch.Tensor):

        return (4 * t * (1 - t)**2) / ((1 + t) ** 2)


    def model_fn(self, model: torch.nn.Module, t: torch.Tensor | float, x:torch.Tensor, aug_cond: torch.Tensor | None):

        if not isinstance(t, torch.Tensor) or t.dim() == 0:
            t = torch.full((x.shape[0],), float(t), dtype=torch.float32, device=self.device)
        elif t.shape[0] != x.shape[0]:
            t = t.expand((x.shape[0], ))

        if self.which == 'simple':

            return model(x, t * 1_000)

        else:
            active_model = getattr(model, "module", model)

            t_emb = timestep_embedding(t * 1_000, self.model_channels)
            emb = active_model.time_embed(t_emb)

            if aug_cond is None:
                aug_cond = torch.zeros((x.shape[0], 9), dtype=torch.float32, device=self.device)

            emb = emb + active_model.aug_proj(aug_cond)

            return model(x, timesteps = None, emb_override=emb)

    def get_model(self):

        self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=self.model_channels, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=2, attention_resolutions=(2,),
                         dropout=0.1,).to(self.device)

        if self.which == 'augmented':
            self.model.aug_proj = torch.nn.Linear(9, self.model_channels * 4, device=self.device)

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
        else:    # self.which == 'augmented':
            x0_aug, aug_cond = self.augmentation_pipeline(x0)

        # [B,]
        t = torch.rand((x0.shape[0],), device=self.device)
        t = torch.clamp(t, min=self.epsilon)

        z = torch.randn_like(x0, device=self.device, dtype=torch.float32)

        ft = self.f(t=t).view(-1, *([1] * (x0.dim() - 1)))
        gt = self.g(t=t).view(-1, *([1] * (x0.dim() - 1)))

        # [B, C, H, W]
        xt = ft * x0_aug + gt * z

        pred_v = self.model_fn(model=model, t=t, x=xt, aug_cond=aug_cond)
        target_v = -x0_aug + 0.5 * gt**(-1) * z

        weight = self.weight(t=t).view(-1, *([1] * (x0_aug.dim() - 1)))

        return (torch.nn.functional.mse_loss(pred_v, target_v, reduction='none') * weight).mean()

    def sample(self, model: torch.nn.Module, amount: int):

        samples = []

        with torch.no_grad():
            for j in range((amount // 512) + 1):
                how_many = min(512, amount - j * 512)

                # [B, C, H, W]
                xT = torch.randn((
                    how_many,
                    self.data.data_dims.channels,
                    self.data.data_dims.height,
                    self.data.data_dims.width,
                ), device=self.device, dtype=torch.float32)
                xt = xT

                dt = (1 - self.epsilon) / self.S
                time_steps = torch.linspace(1, self.epsilon + dt, self.S, device=self.device, dtype=torch.float32)
                for t in time_steps:

                    # [B,]
                    pred_v = self.model_fn(model=model, t=t, x=xt, aug_cond=None)
                    xt = xt - pred_v * dt
        
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

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
    parser.add_argument('--which', type=str, default='simple', choices=['simple', 'augmented'])
    args = parser.parse_args()

    model = DSBFM(which=args.which)
    if args.what == 'full' or args.what == 'train':
        model.train()

    if args.what == 'full' or args.what == 'eval':
        model.eval()