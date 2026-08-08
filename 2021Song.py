import os
import argparse

import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class DDPMppCont():
    """Implements the continuous ddpm model as described in 2021 - Song et al - Score based generative modeling through SDEs
    The prediction target is given by Eq. (7) and we use the probability flow ODE for sampling with a final application of Tweedie's formula.
    
    Their reported 50k FID score:  ~3.25 with 2_000 steps,  and ~3.59 with 1_000 steps
    
    Our achieved 50k FID score with 1_024 steps:    4.02
    Our minimum 2k FID score with 1_024 steps:    27.3
    
    Fewer sampling steps:
        50k fid with S=512:    4.00
        50k fid with S=342:    3.88
        50k fid with S=256:    3.81
        50k fid with S=205:    3.70
        50k fid with S=103:    3.77
        50k fid with S=74:    4.20
        50k fid with S=50:    6.10

        50k fid with S=90:    3.93    <<<    Baseline
        50k fid with S=45:    7.16
        50k fid with S=30:    13.91
        50k fid with S=23:    22.63
        50k fid with S=18:    34.85
        50k fid with S=9:     111.19

    Distillation results
        2x / S=45:
        3x / S=30:
        4x / S=23:
        5x / S=18:
        10x / S=9:

    Thoughts:
        - Since fid score with 205 sampling steps was better than fid score with 1_024 sampling steps, we search even lower. Originally,
                only wanted to have these evaluations for comparison with distilled step counts, but apparently I first have to find the
                baseline.
        - I have to keep in mind that neural networks introduce a second source of errors:    approximation erros. That means that if I
                take too many sampling steps the approximation errors of the network stack up and yield worse results than sampling with
                fewer steps.
        - The training takes like 4 days to finish.
    """

    def __init__(self, S: int = 1_024, load_teacher: bool = False):

        print('Got into the initialization')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.beta0 = 0.1
        self.beta1 = 20

        self.S = S

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

        self.curr_dir = self.base

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

        if load_teacher:

            self.get_model()
            self.model.load_state_dict(torch.load(self.score_save_path, map_location=self.device))

        print(f'Finished initialization.')

    def beta(self, t: torch.Tensor) -> torch.Tensor:

        return self.beta0 + t * (self.beta1 - self.beta0)

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        return - 0.5 * self.beta(t).view(-1, *([1] * (x.dim() - 1))) * x

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        return torch.sqrt(self.beta(t)).view(-1, *([1] * (x.dim() - 1)))

    def b(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        ones = torch.ones(1, device=self.device)
        zeroes = torch.zeros(1, device=self.device)
        return torch.exp(- 0.5 * (t * self.beta(zeroes) + 0.5 * t**2 * (self.beta(ones) - self.beta(zeroes)))).view(-1, *([1] * (x.dim() - 1)))

    def model_fn(self, model: torch.nn.Module, t: torch.Tensor, x: torch.Tensor, aug_cond: None = None) -> torch.Tensor:
        """This outputs the velocity field.    aug_cond exists for uniformity with the other modules."""

        variance = torch.sqrt(1 - self.b(t, x)**2)
        pred_noise = model(x, t)

        return self.f(t, x) - 0.5 * self.g(t, x)**2 * (pred_noise / variance)


    def get_model(self):

        #! the network natively implements group normalization
        self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=128, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=4, attention_resolutions=(2,),
                         dropout=0.1, use_scale_shift_norm=True,
                         resblock_updown=True, use_new_attention_order=True,
                         use_rff=True, rff_scale=16.0).to(self.device)

    def get_ema(self):

        self.ema = torch.optim.swa_utils.AveragedModel(self.model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999))

    def get_optim(self):

        self.optim = torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def noisify(self, t: torch.Tensor, x0: torch.Tensor):

        z = torch.randn_like(x0, device=self.device, dtype=torch.float32)

        mean = self.b(t, x0)
        variance_sq = 1 - mean**2
        variance = torch.sqrt(variance_sq)

        return mean * x0 + variance * z, z

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):
        """The 'score prediction' target algebraically reduces to the noise prediction target."""

        t = torch.rand((x0.shape[0],), device=self.device)
        t = torch.clamp(t, min=1e-5)

        xt, z = self.noisify(t=t, x0=x0)
        
        pred_noise = model(xt, t)
        target_noise = -z

        return torch.nn.functional.mse_loss(pred_noise, target_noise)

    def sample(self, model: torch.nn.Module, amount: int):

        print(f'Sampling with    {self.S}  many steps.')

        samples = []

        with torch.no_grad():
            for i in range((amount // 512) + 1):
                how_many = min(512, amount - i * 512)

                xT = torch.randn((how_many,
                                self.data.data_dims.channels,
                                self.data.data_dims.height,
                                self.data.data_dims.width),
                                device=self.device,
                                dtype=torch.float32)

                dt = (1 - 1e-3) / self.S
                xt = xT
                for j in range(self.S):

                    t = 1 - j * dt
                    t_tensor = torch.full((xt.shape[0],), t, dtype=torch.float32, device=self.device)

                    xt = xt - dt * self.model_fn(model=model, t=t_tensor, x=xt)

                # Final Tweedies application
                final_time = torch.full((xt.shape[0],), 1e-3, dtype=torch.float32, device=self.device)
                final_variance = torch.sqrt(1 - self.b(final_time, xt)**2)
                final_noise_pred = model(xt, final_time)
                xt = (xt + final_variance * final_noise_pred ) / self.b(final_time, xt)


                if amount == 50_000:
                    print(f'sampled {i * 512 + how_many} / 50_000')
                samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    def train(self):

        self.get_model()
        self.get_ema()
        self.get_optim()

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

            self.model.train()
            self.ema.train()

            self.optim.zero_grad()

            loss = self.loss(model=self.model, x0=x0)
            print(f'Loss:  {loss.item()}')
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            print(f'Grad norm: {grad_norm.item()}')

            self.optim.step()
            self.ema.update_parameters(self.model)


            if iteration % 5000 == 0:
                self.ema.eval()
                loss = 0

                with torch.no_grad():
                    for x0, _ in test_dl:
                        x0 = x0.to(self.device)

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

        print('Got into the eval.')
        
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
    parser.add_argument('--S', type=int, default=1_024)
    args = parser.parse_args()

    print('Parsed the args.')

    model = DDPMppCont(S=args.S)
    if args.what == 'full' or args.what == 'train':
        print('Stepped into training clause.')
        model.train()

    if args.what == 'full' or args.what == 'eval':
        print('Stepped into eval clause.')
        model.eval()