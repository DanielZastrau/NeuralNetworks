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
    """Implements the base model as described in 2022 - Salimans Ho - Progressive Distillation

    [30.07.26] - What they didn't mention in the paper is that they use horizontal flipping. I found that in their official implementation
    [30.07.26] - For some reason I had a final integration step after the sampling loop, which I now removed.

    """

    def __init__(self, prediction_target: str = 'v'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prediction_target = prediction_target

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,
            horizontal_flips = True, horizontal_flips_p = 0.5,)
        )

        self.base = '/work/zastrau/Salimans'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.curr_dir = os.path.join(self.base, self.prediction_target)
        if not os.path.exists(self.curr_dir):
            os.mkdir(self.curr_dir)

        self.grid_path = os.path.join(self.curr_dir, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

    def alpha(self, t: torch.Tensor, x: torch.Tensor):
        # ? See the trigonometric formulation in Appendix D

        return torch.cos(0.5 * torch.pi * t).to(self.device).view(-1, *([1] * (x.dim() - 1)))

    def sigma(self, t: torch.Tensor, x: torch.Tensor):
        # ? See the trigonometric formulation in Appendix D

        return torch.sin(0.5 * torch.pi * t).to(self.device).view(-1, *([1] * (x.dim() - 1)))

    def snr(self, t: torch.Tensor, x: torch.Tensor):

        # yields a tensor of shape [B, 1, 1, 1]
        return (self.alpha(t, x)**2 / self.sigma(t, x)**2).to(self.device)

    def snr_trunc(self, t: torch.Tensor, x: torch.Tensor):

        return torch.clamp(self.snr(t, x), min=1.0).to(self.device)

    def snr_pp(self, t: torch.Tensor, x: torch.Tensor):
        """snr + 1"""

        return (self.snr(t, x) + 1).to(self.device)

    def get_model(self):

        return UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=256, channel_mult=(1, 1, 1),
                         num_res_blocks=3, attention_resolutions=(2, 4),
                         dropout=0.2,).to(self.device)

    def get_ema(self, model: torch.nn.Module):

        return torch.optim.swa_utils.AveragedModel(model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999))

    def get_optim(self, model: torch.nn.Module):

        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001) 
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optim,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer=optim,
                                                  start_factor=1e-8,
                                                  end_factor=1.0,
                                                  total_iters=1_000),
                torch.optim.lr_scheduler.ConstantLR(optimizer=optim,
                                                    factor=1.0,
                                                    total_iters=1),
            ],
            milestones=[1_000]
        )

        return optim, scheduler

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        t = torch.rand((x0.shape[0],), device=self.device).clamp(min=1e-5)
        z = torch.randn_like(x0, device=self.device)

        alpha_t = self.alpha(t, x0)
        sigma_t = self.sigma(t, x0)
        weight = self.snr_trunc(t, x0)

        xt = alpha_t * x0 + sigma_t * z

        pred = model(xt, t * 1_000)

        if self.prediction_target == 'x0':

            return (torch.nn.functional.mse_loss(pred, x0, reduction='none') * weight).mean()

        else:    # self.prediction_target == 'v'

            vt = alpha_t * z - sigma_t * x0

            return torch.nn.functional.mse_loss(pred, vt)

    def sample(self, model: torch.nn.Module, amount: int):

        T = 1_000
        dt = 1 / T

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
            for i in range(0, T):

                t = torch.full((xt.shape[0],), 1 - i * dt, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    pred = model(xt, t * 1_000)

                if self.prediction_target == 'x0':
                    alpha_s = self.alpha(t - dt, xt)
                    alpha_t = self.alpha(t, xt)

                    sigma_s = self.sigma(t - dt, xt)
                    sigma_t = self.sigma(t, xt)

                    # we are working with tensors in the range [-1, 1]
                    x0_pred = pred.clamp(min=-1.0, max=1.0)

                    # ? Keep in mind that sigma_t = sin(0.5 * pi * t), for t to zero, this causes a division by increasingly small values
                    # ? But, since sampling is done in discrete time, this is still stable. 
                    xt = alpha_s * x0_pred + sigma_s * ( (xt - alpha_t * x0_pred) / sigma_t)

                else:    # self.prediction_target == 'v':
                    # ? angular update, see appendix d, more stable as it is division free

                    phi_dt = 0.5 * torch.pi * dt
                    xt = torch.cos(torch.tensor(phi_dt)) * xt - torch.sin(torch.tensor(phi_dt)) * pred

            xt = xt.clamp(min=-1.0, max=1.0)
            samples.append(xt.cpu())
            if amount == 50_000:
                print(f'sampled {i * 512 + how_many} / 50_000')

        return torch.cat(samples, dim=0)

    def train(self):

        model = self.get_model()
        ema = self.get_ema(model=model)
        optim, scheduler = self.get_optim(model=model)

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for iteration in range(800_000):
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

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            print(f'Grad norm: {grad_norm.item()}')

            optim.step()
            scheduler.step()
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
    parser.add_argument('--prediction-target', type=str, default='v', choices=['v', 'x0'])
    parser.add_argument('--what', type=str, default='full', choices=['full', 'train', 'eval'])

    args = parser.parse_args()

    Salimans = Diffusion(prediction_target=args.prediction_target)
    if args.what == 'full' or args.what == 'train':
        Salimans.train()

    if args.what == 'full' or args.what == 'eval':
        Salimans.eval()