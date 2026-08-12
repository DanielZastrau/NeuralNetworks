import argparse

import torch

from GenerativeModel import GenerativeModel

from Cluster.utils.dataHandling import DataProvider
from Cluster.utils.dataAugmentation import KarrasAugmentationPipeline
from Cluster.utils.nn_utils import timestep_embedding
from Cluster.utils.sample_kac import TorchKacConstantSampler
from Cluster.utils.velo_utils import compute_velocity
from Cluster.networks.neuralNetworkOpenAI import UNetModel

class Kac(GenerativeModel):
    """2026 - Duong & Chemseddine - Telegraphers Generative Model via Kac Flows
    No point in reimplementing the base model of 2026 - Han et al - DistillKac,
            as it is only a deeper unet.
    
    Later I want to see if I can add some of the diffusion modulations.

    Duong's best 50k FID with S=100:    6.42

    Han's best 50k FID result with S=100:
        Midpoint with guidance strength 1.2:    3.54
        AB2 with guidance strength 1.2:    3.58
        Explicit euler with guidance strengths 1.2 and 1.3:    4.00
    ( NOTE:  Guidance in velocity space is not implemented here. )

    Base model + Euler integrator + uniform schedule
        after 400k iterations
            Best 2k FID with S=100:    28.4
            50k FID with S=100:    4.8
        after 800k iterations
            Best 2k FID with S=100:    28.1
            50k FID with S=100:    5.1
            50k FID with S=50:    7.4
            50k FID with S=34:    14.7
            50k FID with S=25:    26.0
            50k FID with S=20:    38.4
            50k FID with S=10:    101.59
            50k FID with S=5:     201.51

    Distilling 50k FID results x-target:
        Base model 2x:    5.1    >    5.61    <    7.4
        Base model 3x:    5.1    >    5.82    <    14.7
        Base model 4x:    5.1    >    5.79    <    26.0
        Base model 5x:    5.1    >    5.68    <    38.4
        Base model 10x:    5.1    >    123.23    <    101.59

    Distillation 50k FID results v-target:
        Base model 2x:    5.1    >    5.64    <    7.4
        Base model 3x:    5.1    >    5.66    <    14.7
        Base model 4x:    5.1    >    5.68    <    26.0
        Base model 5x:    5.1    >    5.69    <    38.4
        Base model 10x:    5.1    >    
            
    Augmented model + Euler integrator + uniform schedule
        after 400k iterations
            best 2k fid with S=100:    28.71
            best 50k fid with S=100:    5.62
        after 800k iterations
            best 2k fid with S=100:    28.35
            best 50k fid with S=100:    5.09
        after 1.2M iterations
            best 2k fid with S=100:    28.35
            best 50k fid with S=100:    4.91


    Other results
        Base model + Euler integrator + Karras schedule
            50k FID with S=100:    7.1732

        Base model + Heun integrator + uniform schedule
            50k FID with S=50:    9.1535
            50k FID with S=100:    4.8

        Base model + Midpoint integrator + uniform schedule
            50k FID with S=100:    5.79

        Base model + AB2 integrator + uniform schedule
            50k FID with S=100:    5.8

        Base model + Heun integrator + Karras schedule
            50k FID with S=50:    10.8887

    Thoughts:
        - Karras schedule seem to yield worse results
        - Heun integrator also seems to yield worse results
        - Also Euler seems to perform best in general
        - Continued training base 2 for another 400k iterations, because it did not yet converge after the first 400k
        - 2k FID is such a small metric that improvements of just 0.5 or something can also be statistical noise.

    I mistakenly trained my models with time truncation 1e-5, but that difference is marginal, so I am gonna leave it in there.
    """

    def __init__(self, which: str = 'simple', schedule: str = 'uniform',
                 integrator: str = 'euler', S: int = 100,
                 pretrained: bool = False, best_score: float = 10_000.0,
                 load_teacher: bool = False):

        assert schedule in ['uniform', 'karras']
        assert integrator in ['euler', 'heun', 'midpoint', 'ab2']

        super().__init__(base='2026Duong', base_extension=which,
                         S=S, )

        self.which = which

        self.data = DataProvider(args=argparse.Namespace(
            training_batch_size = 128, eval_num_samples = 50_000,
            training_evaluation_period_fid_num_samples = 2_000,)
        )

        self.a = 25    # best according to 2026 - Duong - Telegraphers generative model via Kac flows
        self.c = 2    # best according to 2026 - Duong - Telegraphers generative model via Kac flows

        self.sampler = TorchKacConstantSampler(
            a=self.a,
            c=self.c,
            T=1,
            M=50_000,
            K=4_096,
        )

        self.augmentation_pipeline = KarrasAugmentationPipeline()

        self.T = 1    # max time

        self.schedule = schedule
        self.integrator = integrator

        self.model_channels = 128

        self.karras_p = 7    # staying with the choice of 2022 - Karras - Elucidating the design space of diffusion models

        print(f'The following model is trained:    {self.which}    with integrator:    {self.integrator}    and S:    {self.S}. Pretrained is set to:    {self.pretrained}.')

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
            (self.T**(1/self.karras_p) + (i / (N - 1)) * (self.eps**(1/self.karras_p) - self.T**(1/self.karras_p)))**self.karras_p 
            for i in (range(N))
        ] + [0.0]

        return t_values

    def get_model(self):

        #! The corrected network config of Duong et al
        self.model = UNetModel(image_size=self.data.data_dims.size, in_channels=self.data.data_dims.channels, out_channels=self.data.data_dims.channels,
                         model_channels=self.model_channels, channel_mult=(1, 2, 2, 2),
                         num_res_blocks=2, dropout=0.1,
                         attention_resolutions=(2,), num_heads=4, use_new_attention_order=True,).to(self.device)

        if self.which == 'augmented':
            self.model.aug_proj = torch.nn.Linear(9, self.model_channels * 4, device=self.device).to(self.device)

    def get_optim(self):

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optim,
                                                  start_factor=0.2,
                                                  end_factor=1.0,
                                                  total_iters=self.lr_warmup)

    def get_ema(self):

        self.ema = torch.optim.swa_utils.AveragedModel(self.model, device=self.device, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.9999)).to(self.device)

    def update_ema(self):

        self.ema.update_parameters(self.model)

    def v_fn(self, model: torch.nn.Module, t: torch.Tensor | float, xt: torch.Tensor, aug_cond: torch.Tensor | None = None):
        """This outputs the velocity field."""

        if not isinstance(t, torch.Tensor) or t.dim() == 0:
            t = torch.full((xt.shape[0],), float(t), dtype=torch.float32, device=self.device)
        elif t.shape[0] != xt.shape[0]:
            t = t.expand((xt.shape[0], ))

        if self.which == 'simple':
            return model(xt, t * 1_000)

        else:    # self.which == 'augmented':
            active_model = getattr(model, "module", model)

            t_emb = timestep_embedding(t * 1_000, self.model_channels)
            emb = active_model.time_embed(t_emb)

            if aug_cond is None:
                aug_cond = torch.zeros((xt.shape[0], 9), dtype=torch.float32, device=self.device)

            emb = emb + active_model.aug_proj(aug_cond)

            return model(xt, timesteps = None, emb_override=emb)

    def noisify(self, t: torch.Tensor, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        ft = self.f(t=t)
        gt = self.g(t=t)

        # [B, C x H x W]
        z = self.sampler.sample(gt, dim=self.data.data_dims.total_dimension).to(self.device)

        # [B, C, H, W]
        z = z.reshape(x0.shape)

        # [B, C, H, W] = [B,] * [B, C, H, W] + [B, C, H, W]
        return ft.view(-1, *([1] * (x0.dim() - 1))) * x0 + z, z

    def loss_fn(self, model: torch.nn.Module, x0: torch.Tensor):

        if self.which == 'simple':
            x0_aug, aug_cond = x0, None
        else:    # self.which == 'augmented':
            x0_aug, aug_cond = self.augmentation_pipeline(x0)

        # [B,]
        t = torch.rand((x0.shape[0],), device=self.device)

        # [B,]
        dft = self.df(t=t)
        gt = self.g(t=t)
        dgt = self.dg(t=t)

        xt, z = self.noisify(t=t, x0=x0_aug)

        # [B, C, H, W] = [B,] * [B, C, H, W]
        drift = dft.view(-1, *([1] * (x0.dim() - 1))) * x0_aug
        with torch.no_grad():
            # [B, C, H, W],    retains the shape of z,    shape of x must match shape of t
            velo = dgt.view(-1, *([1] * (x0.dim() - 1))) * compute_velocity(
                x=z,
                t=gt.view(-1, *([1] * (x0.dim() - 1))),
                a=self.a,
                c=torch.tensor(self.c),
                epsilon=self.eps
            )

        # [B, C, H, W]
        pred = self.v_fn(model=model, t=t, xt=xt, aug_cond=aug_cond)

        target = velo + drift

        return torch.nn.functional.mse_loss(pred, target)

    def sample_noise(self, how_many: int) -> torch.Tensor:

        xT = self.sampler.sample(t = torch.ones((how_many,), device=self.device, dtype=torch.float32), dim=self.data.data_dims.total_dimension).to(self.device)
        
        # [B, C, H, W]
        xT = xT.reshape((
            how_many,
            self.data.data_dims.channels,
            self.data.data_dims.height,
            self.data.data_dims.width
        ))

        return xT

    def sample(self, model: torch.nn.Module, amount: int):

        print(f'Sampling with    {self.S}  many steps.')

        samples = []

        with torch.no_grad():
            for j in range((amount // 512) + 1):
                how_many = min(512, amount - j * 512)

                # [B, C x H x W]
                xT = self.sample_noise(how_many=how_many)
                xt = xT

                if self.integrator == 'euler':

                    if self.schedule == 'uniform':
                        dt = (1 - self.eps) / self.S
                        time_steps = torch.linspace(1, self.eps + dt, self.S, device=self.device, dtype=torch.float32)

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
                            dt = t - self.eps

                        pred_v = self.v_fn(model=model, t=t, xt=xt, aug_cond=None)
                        xt = xt - dt * pred_v

                elif self.integrator == 'heun':

                    if self.schedule == 'uniform':
                        dt = (1 - self.eps) / self.S
                        time_steps = torch.linspace(1, self.eps, self.S + 1, device=self.device, dtype=torch.float32)

                    else:    # self.schedule == 'karras'
                        time_steps = torch.tensor(self.get_karras_schedule(self.S))

                    for i in range(len(time_steps) - 1):

                        ti = time_steps[i]
                        tip = time_steps[i + 1]

                        dt = tip - ti

                        # ? Evaluate velocity at ti and take a euler step
                        pred_v_i = self.v_fn(model=model, t=ti, xt=xt, aug_cond=None)
                        x_intermediate = xt + dt * pred_v_i

                        # ? Second order correction
                        if tip != 0:
                            pred_v_ip = self.v_fn(model=model, t=tip, xt=x_intermediate, aug_cond=None)
                            xt = xt + dt * (0.5 * pred_v_i + 0.5 * pred_v_ip)

                        else:
                            xt = x_intermediate

                elif self.integrator == 'midpoint':

                    if self.schedule == 'uniform':
                        time_steps = torch.linspace(1, self.eps, self.S + 1, device=self.device, dtype=torch.float32)
                    else:
                        raise NotImplementedError("Midpoint only implemented for uniform schedule.")

                    for i in range(len(time_steps) - 1):

                        ti = time_steps[i]
                        tip = time_steps[i + 1]
                        dt_step = tip - ti

                        # Evaluate velocity at ti
                        pred_v_i = self.v_fn(model=model, t=ti, xt=xt, aug_cond=None)
                        
                        # Compute the midpoint state
                        t_mid = ti + dt_step / 2
                        x_mid = xt + (dt_step / 2) * pred_v_i
                        
                        # Evaluate velocity at the midpoint
                        pred_v_mid = self.v_fn(model=model, t=t_mid, xt=x_mid, aug_cond=None)
                        
                        # Take the full step using the midpoint velocity
                        xt = xt + dt_step * pred_v_mid

                elif self.integrator == 'ab2':
                    # Only implement for uniform as requested
                    if self.schedule == 'uniform':
                        time_steps = torch.linspace(1, self.eps, self.S + 1, device=self.device, dtype=torch.float32)
                    else:
                        raise NotImplementedError("AB2 only implemented for uniform schedule.")

                    pred_v_prev = None
                    for i in range(len(time_steps) - 1):
                        ti = time_steps[i]
                        tip = time_steps[i + 1]
                        dt_step = tip - ti

                        # Evaluate velocity at ti
                        pred_v_i = self.v_fn(model=model, t=ti, xt=xt, aug_cond=None)
                        
                        if i == 0:
                            # AB2 requires a history of 1 step; use standard Euler for the initial step
                            xt = xt + dt_step * pred_v_i
                        else:
                            # Adams-Bashforth 2 explicit step
                            xt = xt + dt_step * (1.5 * pred_v_i - 0.5 * pred_v_prev)
                            
                        # Store current velocity for the next step's history 
                        pred_v_prev = pred_v_i

                if amount == 50_000:
                    print(f'sampled {j * 512 + how_many} / 50_000')
                samples.append(xt.cpu())

        return torch.cat(samples, dim=0)

    def extract_gradient_norm(self) -> torch.Tensor:
        return torch.sqrt(sum(p.grad.data.norm() ** 2 for p in self.model.parameters() if p.grad is not None))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--what', type=str, choices=['full', 'train', 'eval'], default='full')
    parser.add_argument('--which', type=str, default='simple', choices=['simple', 'augmented'])
    parser.add_argument('--schedule', type=str, default='uniform', choices=['uniform', 'karras'])
    parser.add_argument('--integrator', type=str, default='euler', choices=['euler', 'heun', 'midpoint', 'ab2'])
    parser.add_argument('--S', type=int, default=100)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--best-score', type=float, help='If resuming training of a pretrained model, provide its best score')
    args = parser.parse_args()

    if args.pretrained: assert args.best_score is not None

    model = Kac(which=args.which, schedule=args.schedule,
                integrator=args.integrator, S=args.S,
                pretrained=args.pretrained, best_score=args.best_score if args.best_score is not None else 10_000.0)
    
    if args.what == 'full' or args.what == 'train':
        model.train()

    if args.what == 'full' or args.what == 'eval':
        model.eval()