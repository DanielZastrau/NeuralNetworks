import os
import time

from abc import abstractmethod

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb

class GenerativeModel():

    def __init__(self, base: str, base_extension: str,
                 I: int = 400_000, S: int = 1_024,
                 lr: float = 2e-4, lr_warmup_factor: float = 0.05, eps: float = 1e-5,
                 pretrained: bool = False, load_teacher: bool = False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data: DataProvider

        self.pretrained: bool = pretrained
        self.model: torch.nn.Module
        self.ema: torch.optim.swa_utils.AveragedModel
        self.optim: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LRScheduler

        self.I: int = I    # training iterations
        self.S: int = S    # sampling steps

        self.eps: float = eps
        self.lr: float = lr    # learning rate
        self.lr_warmup = int(self.I * lr_warmup_factor)

        self.base = f'/work/zastrau/{base}'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.curr_dir = os.path.join(self.base, base_extension)
        if not os.path.exists(self.curr_dir):
            os.mkdir(self.curr_dir)

        self.grid_path = os.path.join(self.curr_dir, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.curr_dir, 'best_score_model.pth')

        if load_teacher:

            self.get_model()
            self.model.load_state_dict(torch.load(self.score_save_path, map_location=self.device))

    @abstractmethod
    def get_model(self):
        ...

    @abstractmethod
    def get_optim(self):
        ...

    @abstractmethod
    def get_ema(self):
        ...

    @abstractmethod
    def update_ema(self):
        ...

    @abstractmethod
    def v_fn(self, model: torch.nn.Module, t: torch.Tensor, xt: torch.Tensor, aug_cond: torch.Tensor | None = None) -> torch.Tensor:
        """This returns the velocity field in a form that the Euler step can be directly executed.
        xt = xt - dt * vt"""
        ...

    @abstractmethod
    def noisify(self, t: torch.Tensor, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Corrupt a clean data sample to a time point t."""
        ...

    @abstractmethod
    def loss_fn(self, model: torch.nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """Implements the loss target of the model."""
        ...

    @abstractmethod
    def sample_noise(self, how_many: int) -> torch.Tensor:
        """Yields the initial noise for the sampling process.
        Output shape should be [B, C, H, W]"""
        ...

    def sample(self, model: torch.nn.Module, amount: int) -> torch.Tensor:
        """Implements the sampling loop of the model.
        This only implements a basic uniform schedule and Euler integrator.
        For anything else, this methods has to be overwritten."""

        print(f'Sampling with    {self.S}  steps.')

        samples = []

        dt = (1 - self.eps) / self.S
        time_steps = torch.linspace(1, self.eps + dt, self.S, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for i in range((amount // 512) + 1):
                how_many = min(512, amount - i * 512)

                xT = self.sample_noise(how_many=how_many)
                xt = xT.clone()

                for j in range(len(time_steps)):
                    t = time_steps[j]

                    vt = self.v_fn(model=model, t=t, xt=xt, aug_cond=None)
                    xt = xt - dt * vt

                if amount == 50_000:
                    print(f'Sampled {i * 512 + how_many}  /  50 000.')

                samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    @abstractmethod
    def extract_gradient_norm(self) -> torch.Tensor:
        """Needs to return the gradient norm. Can either just calculate it or it implements gradient clipping."""
        ...

    def train(self):

        self.get_model()

        if self.pretrained:
            self.model.load_state_dict(torch.load(self.score_save_path, map_location=self.device))

        self.get_ema()
        self.get_optim()

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for iteration in range(self.I):
            if iteration % 1_000 == 0:
                print(f'----------    Iteration    {iteration}    ----------')

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

            loss = self.loss_fn(model=self.model, x0=x0)
            print(f'Loss:  {loss.item()}.')
            loss.backward()

            grad_norm = self.extract_gradient_norm()
            print(f'Grad norm: {grad_norm.item()}.')

            self.optim.step()
            self.scheduler.step()
            self.update_ema()

            if (iteration + 1) % 5000 == 0:
                self.ema.eval()
                loss = 0

                with torch.no_grad():
                    for x0, _ in test_dl:
                        x0 = x0.to(self.device, dtype=torch.float32)

                        loss += self.loss_fn(model=self.ema, x0=x0).detach()

                avg_loss = loss.item() / len(test_dl)
                print(f'Average test loss:    {avg_loss}.')


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
    
                print(f'Generated an 8x8 grid and saved it to:  {self.grid_path}.')

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
                    print(f"Saved best score model to:  {self.score_save_path}.")

    def eval(self):
                    
        # ! Final Fid evaluation on 50_000 samples

        t = time.time()

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
        print(f"It took    {(time.time() - t) / 60}  minutes.")