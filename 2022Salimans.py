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
    [01.08.26] - Worked through the official implementation. They first define a numerically stable log snr and then redefine everything on
                    top of it.
    [03.08.26] - Still does not produce results, going to implement their work even further.
                    Their implementation first calculates all entities from the network predicition and then calculates all losses x0, z, v.
                    From there it routes to the correct loss based on the weighting wanted, i.e.,  constant weight to x0,  snr weight to z,
                    snr_trunc weight to the maximum between x0 and z,  and snrpp to v

    This implements:
        - x0 prediction with snr_trunc weighting and a { stable log snr schedule (adapted from their official repo) } with direct DDIM sampling
        - v prediction in angular coordinates and angular DDIM sampling (see appendix D)

    Their Fid scores:
        - snr_trunc weighted x0-pred and 512 sampling steps     2.51
        - unweighted v-prediction with 512 sampling steps     2.87
    """

    def __init__(self, prediction_target: str = 'x0', loss_target: str = 'x0'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prediction_target = prediction_target
        self.loss_target = loss_target

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,
            horizontal_flips = True, horizontal_flips_p = 0.5,)
        )

        self.log_snr_min = -20
        self.log_snr_max = 20

        self.I = 800_000    # amount of training iterations
        self.S = 512    # amount of sampling steps

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

    def stable_log_snr(self, t: torch.Tensor):
        """Follows the true log snr on the itnerior, but is capped by the min max values at the boundaries"""

        b = torch.arctan(torch.exp(torch.tensor(-0.5 * self.log_snr_max, device=self.device)))
        a = torch.arctan(torch.exp(torch.tensor(-0.5 * self.log_snr_min, device=self.device))) - b

        return - 2 * torch.log( torch.tan(a * t + b))

    def stable_snr_trunc(self, t: torch.Tensor, x: torch.Tensor):
        """See section 4. max(alpha_t^2 / sigma_t^2), 1"""

        snr = torch.exp(self.stable_log_snr(t=t))
        ones = torch.ones_like(t, device=self.device, dtype=torch.float32)

        return torch.maximum(snr, ones).view(-1, *([1] * (x.dim() - 1)))

    def alpha_snr(self, t: torch.Tensor, x: torch.Tensor):
        """In appendix A they provide a reformulation of alpha in terms of the sigmoid function"""

        lambda_t = self.stable_log_snr(t=t)
        return torch.sqrt(torch.sigmoid(lambda_t)).view(-1, *([1] * (x.dim() - 1)))

    def sigma_snr(self, t: torch.Tensor, x: torch.Tensor):
        """This is not explicitely stated in the appendix, but follows from alpha"""

        lambda_t = self.stable_log_snr(t=t)
        return torch.sqrt(torch.sigmoid(- lambda_t)).view(-1, *([1] * (x.dim() - 1)))

    def predict_x0_from_z(self, xt: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        """x = (xt - sigma z) / alpha"""

        alpha_t = self.alpha_snr(t=t, x=xt)
        sigma_t = self.sigma_snr(t=t, x=xt)

        return (xt - sigma_t * z) / alpha_t

    def predict_z_from_x0(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor):
        """z = (xt - alpha x ) / sigma"""

        alpha_t = self.alpha_snr(t=t, x=xt)
        sigma_t = self.sigma_snr(t=t, x=xt)
        return (xt - alpha_t * x0) / sigma_t

    def predict_x0_from_v(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """x0 = alpha * xt - sigma * v
        
        Follows from the trigonometric definitions."""

        alpha_t = self.alpha_snr(t=t, x=xt)
        sigma_t = self.sigma_snr(t=t, x=xt)

        return alpha_t * xt - sigma_t * v

    def predict_v_from_x0_and_z(self, x0: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        """v = alpha z - sigma x0"""

        alpha_t = self.alpha_snr(t=t, x=x0)
        sigma_t = self.sigma_snr(t=t, x=x0)

        return alpha_t * z - sigma_t * x0

    def get_model(self):

        return UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=256, channel_mult=(1, 1, 1),
                         num_res_blocks=3, attention_resolutions=(2, 4),
                         dropout=0.2,).to(self.device)

    def get_ema(self, model: torch.nn.Module):

        return torch.optim.swa_utils.AveragedModel(model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999))

    def get_optim(self, model: torch.nn.Module):

        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001) 
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optim,
                                                  start_factor=1e-8,
                                                  end_factor=1.0,
                                                  total_iters=1_000)

        return optim, scheduler

    def loss(self, model: torch.nn.Module, x0: torch.Tensor):

        t = torch.rand((x0.shape[0],), device=self.device)
        z = torch.randn_like(x0, device=self.device)

        alpha_t = self.alpha_snr(t=t, x=x0)
        sigma_t = self.sigma_snr(t=t, x=x0)

        xt = alpha_t * x0 + sigma_t * z

        pred = model(xt, t * 1_000)

        # Calculate the other entities from the prediction.  ( as was done in the original implementation )
        if self.prediction_target == 'x0':

            pred_x0 = pred

            pred_z = self.predict_z_from_x0(xt=xt, x0=pred_x0, t=t)
            pred_v = self.predict_v_from_x0_and_z(x0=pred_x0, z=pred_z, t=t)

        else:    # self.prediction_target == 'v':

            pred_v = pred

            pred_x0 = self.predict_x0_from_v(xt=xt, v=pred_v, t=t)
            pred_z = self.predict_z_from_x0(xt=xt, x0=pred_x0, t=t)

        pred_x0 = pred_x0.clamp(min=-1.0, max=1.0)

        x0_target = x0
        z_target = z
        v_target = self.predict_v_from_x0_and_z(x0=x0, z=z, t=t)

        # Compute the per-sample loss
        x0_loss = torch.nn.functional.mse_loss(pred_x0, x0_target, reduction='none').mean(dim=[1, 2, 3])
        z_loss = torch.nn.functional.mse_loss(pred_z, z_target, reduction='none').mean(dim=[1, 2, 3])
        v_loss = torch.nn.functional.mse_loss(pred_v, v_target, reduction='none').mean(dim=[1, 2, 3])

        if self.loss_target == 'x0':    #! this is constant weighting  (not referenced in the paper)
            return x0_loss.mean()
        elif self.loss_target == 'z':    #! this is the snr weighting
            return z_loss.mean()
        elif self.loss_target == 'v':    #! this is the snrpp weighting
            return v_loss.mean()
        else:    # self.loss_target == 'x0z':    #! this is the snr_trunc weighting
            return torch.maximum(x0_loss, z_loss).mean()


    def sample(self, model: torch.nn.Module, amount: int):

        dt = 1 / self.S

        samples = []

        for j in range((amount // 512) + 1):
            how_many = min(512, amount - j * 512)

            xT = torch.randn((how_many,
                              self.data.data_dims.channels,
                              self.data.data_dims.height,
                              self.data.data_dims.width),
                              device=self.device,
                              dtype=torch.float32)

            xt = xT
            for i in range(0, self.S):

                t = torch.full((xt.shape[0],), 1 - i * dt, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    pred = model(xt, t * 1_000)

                alpha_s = self.alpha_snr(t=t - dt, x=xt)
                sigma_s = self.sigma_snr(t=t - dt, x=xt)

                # Calculate the other entities from the prediction.  ( as was done in the original implementation )
                if self.prediction_target == 'x0':

                    pred_x0 = pred.clamp(min=-1.0, max=1.0)
                    pred_z = self.predict_z_from_x0(xt=xt, x0=pred_x0, t=t)

                else:    # self.prediction_target == 'v':

                    pred_x0 = self.predict_x0_from_v(xt=xt, v=pred, t=t).clamp(min=-1.0, max=1.0)
                    pred_z = self.predict_z_from_x0(xt=xt, x0=pred_x0, t=t)

                xt = alpha_s * pred_x0 + sigma_s * pred_z

            xt = xt.clamp(min=-1.0, max=1.0)
            samples.append(xt.cpu())
            if amount == 50_000:
                print(f'sampled {j * 512 + how_many} / 50_000')

        return torch.cat(samples, dim=0)

    def train(self):

        model = self.get_model()
        ema = self.get_ema(model=model)
        optim, scheduler = self.get_optim(model=model)

        train_dl, test_dl = self.data.get_datasets_for_training()
        eval_dl = self.data.get_dataset_for_periodic_eval()

        train_iter = iter(train_dl)
        for iteration in range(self.I):
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
    parser.add_argument('--what', type=str, default='full', choices=['full', 'train', 'eval'])
    parser.add_argument('--prediction-target', type=str, default='x0', choices=['v', 'x0'])
    parser.add_argument('--loss-target', type=str, default='x0', choices=['x0', 'z', 'v', 'x0z'])
    args = parser.parse_args()

    Salimans = Diffusion(prediction_target=args.prediction_target)
    if args.what == 'full' or args.what == 'train':
        Salimans.train()

    if args.what == 'full' or args.what == 'eval':
        Salimans.eval()