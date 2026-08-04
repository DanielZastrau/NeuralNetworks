import os
import argparse
import math
import copy

import lpips
import torch
import torchvision
import torch_fidelity
import matplotlib.pyplot as plt

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.uint8_utils import Uint8Dataset, to_uint8_rgb
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class ConsistencyModel():
    """2023 - Song et al. - Consistency Models

    [30.07.26] - Ahh okay OpenAI publishes a model, which only becomes viable once you are rich. who would have thought.
                    Training the model on Cifar-10 with 1 GPU takes about ~200 - ~220 hours. That is why they trained
                    on 8 GPUS and their larger models for LSUN for example on 64 GPUs.
    [30.07.26] - I will just let the task run to completion in the 144 hours I alloted to it and then use whatever best
                    FID it achieved. Maybe in the future I will tackle implementing the distributed version on multiple GPUs.
    [30.07.26] - I just noticed that I had a bright moment when starting the job. Taking the hint from the 8 GPUs they used,
                    I was cautious enough to just set the time limit to 168 hours. That should allow it to get through roughly
                    640k iterations.
    [04.08.26] - Stopped training. This did not produce satisfying results and I saw no further purpose to trying to fix it.
                    Neither was I still learning something by trying to fix it nor was it needed for my master thesis.
                    My continue in the future at some point.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,)
        )

        self.T = 80
        self.K = 800_000
        self.S = 50
        self.s0 = 2
        self.s1 = 150
        self.mu0 = 0.9

        self.time_factor = 1_000 / self.T
        self.epsilon = 0.002
        self.karras_p = 7.0

        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)

        self.base = '/work/zastrau/2023Song'
        if not os.path.exists(self.base):
            os.mkdir(self.base)

        self.grid_path = os.path.join(self.base, 'grids')
        if not os.path.exists(self.grid_path):
            os.mkdir(self.grid_path)

        self.best_score = 10_000.0
        self.score_save_path = os.path.join(self.base, 'best_score_model.pth')

    def c_one(self, t: torch.Tensor) -> torch.Tensor:

        return (0.5)**2 / ((t - self.epsilon)**2 + 0.5**2)

    def c_two(self, t: torch.Tensor) -> torch.Tensor:

        return (0.5 * (t - self.epsilon)) / (torch.sqrt(0.5**2 + t**2))

    def f(self, model: torch.nn.Module, t: torch.Tensor, x: torch.Tensor):

        return self.c_one(t).view(-1, *([1] * (x.dim() - 1))) * x + self.c_two(t).view(-1, *([1] * (x.dim() - 1))) * model(x, t * self.time_factor)

    def N_fn(self, k: int):

        factor = k / self.K
        inner_bracket = self.s1 + 1
        outer_bracket = inner_bracket**2 - self.s0**2
        sqrt = math.sqrt(factor * outer_bracket + self.s0**2)
        ceil = math.ceil(sqrt - 1)
        return ceil + 1

    def mu_fn(self, k: int):

        return math.exp(self.s0 * math.log(self.mu0) / self.N_fn(k))

    def get_karras_schedule(self, k: int = 0, training: bool = True):
        """Generates the N(k)-discretized time steps for iteration k."""
        time_steps = []
        epsilon_root = self.epsilon ** (1.0 / self.karras_p)
        T_root = self.T ** (1.0 / self.karras_p)

        if training:
            for i in range(self.N_fn(k)):
                factor = i / (self.N_fn(k) - 1)
                time_steps.append((epsilon_root + factor * (T_root - epsilon_root)) ** self.karras_p)

            return torch.tensor(time_steps, device=self.device)
        else:    # sampling
            for i in reversed(range(self.S - 1)):
                factor = i / (self.S - 1)
                time_steps.append((epsilon_root + factor * (T_root - epsilon_root)) ** self.karras_p)
            
            return time_steps

    def get_model(self):

        #! the network natively implements group normalization
        return UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=128, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=4, attention_resolutions=(2,),
                         dropout=0.0, use_scale_shift_norm=True,
                         resblock_updown=True, use_new_attention_order=True,).to(self.device)

    def get_optim(self, model: torch.nn.Module):

        return torch.optim.RAdam(model.parameters(), lr=4e-4)

    def loss(self, model: torch.nn.Module, target_model: torch.nn.Module, x0: torch.Tensor, k: int):
        karras_schedule = self.get_karras_schedule(k=k)

        n = torch.randint(low=0, high=self.N_fn(k) - 1, size=(x0.shape[0],), device=self.device)
        tn = karras_schedule[n]
        tnp = karras_schedule[n + 1]
        z = torch.randn_like(x0, device=self.device)

        xt = x0 + tn.view(-1, *([1] * (x0.dim() - 1))) * z
        xtp = x0 + tnp.view(-1, *([1] * (x0.dim() - 1))) * z

        prediction_one = self.f(model=model, t=tnp, x=xtp)

        with torch.no_grad():
            prediction_two = self.f(model=target_model, t=tn, x=xt)

        # Upsample to 224x224 as specified in Appendix C of Song et al. (2023)
        pred_one_224 = torch.nn.functional.interpolate(prediction_one, size=(224, 224), mode='bilinear', align_corners=False)
        pred_two_224 = torch.nn.functional.interpolate(prediction_two, size=(224, 224), mode='bilinear', align_corners=False)

        # Calculate LPIPS. The module returns shape (N, 1, 1, 1), so take the mean.
        return self.lpips_loss(pred_one_224, pred_two_224).mean()


    def sample(self, model: torch.nn.Module, amount: int):

        samples = []

        with torch.no_grad():
            for i in range((amount // 512) + 1):
                how_many = min(512, amount - i * 512)

                xT = torch.randn((how_many,
                                self.data.data_dims.channels,
                                self.data.data_dims.height,
                                self.data.data_dims.width),
                                device=self.device,
                                dtype=torch.float32) * self.T
                t_initial = torch.full((xT.shape[0],), self.T, dtype=torch.float32, device=self.device)
                xt = self.f(model=model, t=t_initial, x=xT)

                time_steps = self.get_karras_schedule(training=False)

                for t in time_steps:

                    t_tensor = torch.full((xT.shape[0],), t, dtype=torch.float32, device=self.device)
                    z = torch.randn_like(xT, device=self.device)

                    xt = xt + torch.sqrt(torch.tensor(t**2 - self.epsilon**2, device=self.device)) * z
                    xt = self.f(model=model, t=t_tensor, x=xt)

                if amount == 50_000:
                    print(f'sampled {i * 512 + how_many} / 50_000')
                samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    def train(self):

        model = self.get_model()
        target_model = copy.deepcopy(model).to(self.device)
        for param in target_model.parameters():
            param.requires_grad = False
        target_model.eval()

        optim = self.get_optim(model=model)

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for k in range(self.K):
            if k % 1_000 == 0:
                print(f'----------    iteration    {k}    ----------')

            try:
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)
            except StopIteration:
                train_iter = iter(train_dl)
                x0, _ = next(train_iter)
                x0 = x0.to(self.device, dtype=torch.float32)

            model.train()

            optim.zero_grad()

            loss = self.loss(model=model, target_model=target_model, x0=x0, k=k)
            print(f'Loss:  {loss.item()}')

            loss.backward()

            grad_norm = torch.sqrt(sum(p.grad.data.norm() ** 2 for p in model.parameters() if p.grad is not None))
            print(f'Grad norm: {grad_norm.item()}')

            optim.step()

            mu = self.mu_fn(k=k)
            with torch.no_grad():
                for p_target, p_online in zip(target_model.parameters(), model.parameters()):
                    p_target.mul_(mu).add_(p_online, alpha=1.0 - mu)


            if k % 5000 == 0:
                target_model.eval()
                loss = 0

                with torch.no_grad():
                    for x0, _ in test_dl:
                        x0 = x0.to(self.device)

                        loss += self.loss(model=target_model, target_model=target_model, x0=x0, k=k).detach()

                avg_loss = loss.item() / len(test_dl)
                print(f'>>>>>>>>>> avg test loss:    {avg_loss}')


            # regularly sample a small grid to check progress
            if (k + 1) % 10_000 == 0:
                target_model.eval()
    
                samples = self.sample(model=target_model, amount=64)    # are [-1, 1]
                samples = (samples + 1.0) * 0.5    # now [0, 1]
                samples = samples.clamp(0.0, 1.0)    # for good measure

                grid = torchvision.utils.make_grid(samples, nrow=8, padding=2, normalize=False)
                plt.figure(figsize=(8, 8))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                plt.axis("off")
        
                plt.savefig(os.path.join(self.grid_path, f'{k}.png'), dpi=200, bbox_inches="tight", pad_inches=0)
                
                plt.close()
    
                print(f'-----------------------------------------------generated an 8x8 grid and saved it to:  {self.grid_path}')

            if (k + 1) % 50_000 == 0:

                target_model.eval()
                samples = self.sample(model=target_model, amount=2_000)

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

                    model_to_save = getattr(target_model, "module", target_model)
                    uncompiled_model = getattr(model_to_save, "_orig_mod", model_to_save)
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

    CM = ConsistencyModel()
    if args.what == 'full' or args.what == 'train':
        CM.train()

    if args.what == 'full' or args.what == 'eval':
        CM.eval()