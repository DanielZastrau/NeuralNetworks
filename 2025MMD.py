import os
import argparse

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.dataAugmentation import KarrasAugmentationPipeline
from Cluster.utils.nn_utils import timestep_embedding
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class MMD():
    """The MMD Gradient flow toward a uniform distribution was talked about (introduced in?)
    2026 - Chemseddine et al - Adapting Noise to Data

    Added a config for Karras augmentation. Currently it is only added on top of the smaller model.

    With the smaller model and 400k iterations:
        Min 2k FID with S=1_024:    ~29.04
        50k FID with S=1_024:    ~6.4
    
    With smaller model, data augmentation and 600k iterations:
        Min 2k FID with S=1_024:    ~29.5
        50k FID with S=1_024:    ~6.3
        
        50k fid with S=512:    ~6.6    <<<    Baseline
        50k fid with S=256:    ~7.41
        50k fid with S=171:    ~8.71
        50k fid with S=128:    ~10.32
        50k fid with S=103:    ~12.46
        50k fid with S=52:    ~26.14

    Distillation results:
        50k fid with 2x distill / S=256:
        50k fid with 3x distill / S=171:
        50k fid with 4x distill / S=128:
        50k fid with 5x distill / S=103:
        50k fid with 10x distill / S=52:

    With the bigger model:
        Min 2k FID with S=1_024:    ~28    I was stupid and overwrote the output file with a new job, before I could make the grid...
        Min 50k FID with S=1_024:    5.4
    """

    def __init__(self, size: str = 'small', data_augmentation: bool = False, I: int = 400_000, S: int = 1_024, load_teacher: bool = False):

        print(f'The following model is trained:    {size}, with data augmentation?    {data_augmentation}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,)
        )

        self.augmented = data_augmentation
        if self.augmented:
            self.augmentation_pipeline = KarrasAugmentationPipeline()

        self.size = size

        self.b = 3

        self.I = I
        self.lr = 2e-4
        self.lr_warmup = int(self.I * 0.05)
        self.epsilon = 1e-5
        self.S = S    # amount of sampling steps

        self.model_channels = 128

        self.base = '/work/zastrau/2025MMD'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        s = f'{self.size}augmented' if self.augmented else f'{size}'
        self.curr_dir = os.path.join(self.base, s)
        if not os.path.exists(self.curr_dir):
            os.mkdir(self.curr_dir)

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

        if load_teacher:

            self.get_model()
            self.model.load_state_dict(torch.load(self.score_save_path, map_location=self.device))


    def f(self, t: torch.Tensor):

        return 1 - t

    def df(self, t: torch.Tensor):

        return torch.ones_like(t, device=self.device) * -1

    def g(self, t: torch.Tensor):

        return t

    def dg(self, t: torch.Tensor):

        return torch.ones_like(t, device=self.device)

    def noise(self, t: torch.Tensor, x: torch.Tensor):

        # [B, C, H, W]
        # Sample uniform noise and normalize to [-1, 1]
        pre_z = torch.rand_like(x, device=self.device, dtype=torch.float32) * 2 - 1

        # [B, C, H, W]
        factor = (1 - torch.exp(- t / torch.tensor(self.b))).view(-1, *([1] * (x.dim() - 1)))
        z = self.b * factor * pre_z

        return pre_z, z

    def model_fn(self, model: torch.nn.Module, t: torch.Tensor | float, x:torch.Tensor, aug_cond: torch.Tensor | None):

        if not isinstance(t, torch.Tensor) or t.dim() == 0:
            t = torch.full((x.shape[0],), float(t), dtype=torch.float32, device=self.device)
        elif t.shape[0] != x.shape[0]:
            t = t.expand((x.shape[0], ))

        if not self.augmented:
            return model(x, t * 1_000)

        else:    # self.size == 'augmented':
            active_model = getattr(model, "module", model)

            t_emb = timestep_embedding(t * 1_000, self.model_channels)
            emb = active_model.time_embed(t_emb)

            if aug_cond is None:
                aug_cond = torch.zeros((x.shape[0], 9), dtype=torch.float32, device=self.device)

            emb = emb + active_model.aug_proj(aug_cond)

            return model(x, timesteps = None, emb_override=emb)

    def get_model(self):

        if self.size == 'small':
            self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                            model_channels=self.model_channels, channel_mult=(1, 2, 2, 2),
                            num_res_blocks=2, attention_resolutions=(2,),
                            dropout=0.1,).to(self.device)

        elif self.size == 'medium':
            self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                            model_channels=self.model_channels, channel_mult=(1, 2, 2, 2),
                            num_res_blocks=3, attention_resolutions=(2, 4),
                            dropout=0.1,).to(self.device)

        if self.augmented:
            self.model.aug_proj = torch.nn.Linear(9, self.model_channels * 4, device=self.device).to(self.device)
        

    def get_ema(self):

        self.ema = torch.optim.swa_utils.AveragedModel(self.model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999)).to(self.device)

    def get_optim(self):

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optim,
                                                  start_factor=0.2,
                                                  end_factor=1.0,
                                                  total_iters=self.lr_warmup)

    def noisify(self, t: torch.Tensor, x0: torch.Tensor):

        ft = self.f(t=t).view(-1, *([1] * (x0.dim() - 1)))

        # [B, C, H, W]
        pre_z, z = self.noise(t=t, x=x0)

        # [B, C, H, W]
        xt = ft * x0 + z

        return xt, pre_z

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        if self.augmented:
            x0_aug, aug_cond = self.augmentation_pipeline(x0)
        else:
            x0_aug, aug_cond = x0, None

        # [B,]
        t = torch.rand((x0.shape[0],), device=self.device)

        # [B,]
        dft = self.df(t=t).view(-1, *([1] * (x0.dim() - 1)))
        gt = self.g(t=t).view(-1, *([1] * (x0.dim() - 1)))
        dgt = self.dg(t=t).view(-1, *([1] * (x0.dim() - 1)))

        xt, pre_z = self.noisify(t=t, x0=x0_aug)

        pred = self.model_fn(model=model, t=t, x=xt, aug_cond=aug_cond)
        target = dft * x0_aug + dgt * (pre_z / torch.exp(gt / self.b))

        return torch.nn.functional.mse_loss(pred, target)

    def sample(self, model: torch.nn.Module, amount: int):

        samples = []

        with torch.no_grad():
            for j in range((amount // 512) + 1):
                how_many = min(512, amount - j * 512)

                # [B, C, H, W]
                shape_tensor = torch.ones((
                    how_many,
                    self.data.data_dims.channels,
                    self.data.data_dims.height,
                    self.data.data_dims.width
                ), device=self.device)
                _, xT = self.noise(t=torch.ones((how_many,), device=self.device, dtype=torch.float32), x=shape_tensor)
                xt = xT

                dt = (1 - self.epsilon) / self.S
                time_steps = torch.linspace(1, self.epsilon + dt, self.S, device=self.device, dtype=torch.float32)
                for t in time_steps:

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
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium'])
    parser.add_argument('--data-augmentation', action='store_true')
    parser.add_argument('--I', type=int, default=400_000)
    parser.add_argument('--S', type=int, default=1_024)
    args = parser.parse_args()

    model = MMD(size = args.size, data_augmentation=args.data_augmentation, I=args.I, S=args.S)
    if args.what == 'full' or args.what == 'train':
        model.train()

    if args.what == 'full' or args.what == 'eval':
        model.eval()