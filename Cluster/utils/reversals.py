import argparse

import torch

from Cluster.utils.dataHandling import DataProvider


class Reversal():
    """This class implements the different reversal methods for the models, it is called by the sampling module as well as the distillation module
    
    All methods should get a starting time point t, and either a delta_t or t_end, a model and a data point x.
    """
    # ! If used in the distillation module, only explicit methods with fixed timesteps are to be specified, since distillation does not exactly work with adaptive solvers.
    # TODO check that intuition again

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Route the teacher integration method
        if args.which == 'diffusion':
            if args.distill_teacher_sampler == 'sde':
                self.teacher_integrate = self.diffusion_euler_maruyama
            else:    
                # pfODE relies on adaptive scipy solvers not suited for fixed-step distillation
                raise NotImplementedError("Distillation requires explicit fixed-step solvers. Your current config wants to use rk45 with the pfODE for diffusion")
        elif args.which == 'kac':
            if args.distill_teacher_sampler == 'ee':
                self.teacher_integrate = self.kac_explicit_euler
            elif args.distill_teacher_sampler == 'rk2':
                self.teacher_integrate = self.kac_rk2
            else:
                raise NotImplementedError("Use 'ee' or 'rk2' for Kac distillation.")

        # Route the student integration method
        # Both currently share the explicit Euler step structure from Han et al. 2025
        self.student_integrate = self.student_explicit_euler


    # =============================================================================================
    # student solver

    def student_explicit_euler(self, model: torch.nn.Module, x_batch: torch.Tensor, 
                               t_batch: torch.Tensor, dt: float) -> torch.Tensor:
        """Integrates the student over [t, t - dt] using a single explicit step."""
        return x_batch - model(x_batch, t_batch) * dt


    # =============================================================================================
    # general wrapper

    def rk45_wrapper(self, model: torch.nn.Module, data: DataProvider, x_batch: torch.Tensor, t_start: float, t_end: float) -> torch.Tensor:
        """RK45 solver for the flow ODE
        
        It is only used in the sampling module on [self.args.T, t_end]
        where t_end = self.args.time_truncation for diffusion and 0 for kac
        
        Cannot be used in distillation since distillation needs a fixed timestep solver"""

        from torchdiffeq import odeint

        device = next(model.parameters()).device

        if self.args.which == 'diffusion':
            assert t_start == self.args.T
            assert t_end == self.args.time_truncation
            
            ode_fn = self.DiffusionODEDerivative(model=model, min_t=self.args.time_truncation).to(device)

        else:    # self.args.which == 'kac'
            assert t_start == self.args.T
            assert t_end == 0

            ode_fn = self.KacODEDerivative(
                model=model, 
                channels=data.data_dims.channels, 
                width=data.data_dims.width, 
                height=data.data_dims.height
            ).to(device)

        # Pass only the start and end points to prevent intermediate VRAM caching
        t_vals = torch.tensor([self.args.T, t_end], device=device)

        with torch.no_grad():
            sol = odeint(
                ode_fn, 
                x_batch, 
                t_vals, 
                method='dopri5', 
                rtol=self.args.rel_tol, 
                atol=self.args.abs_tol
            )

            print(f'It took  {ode_fn.nfe}  nfes to complete with a{self.args.abs_tol}  r{self.args.rel_tol}')

        # Extract terminal state
        return sol[-1]

    
    # =============================================================================================
    # kac explicit solver implementations


    # ! Self implement these methods to avoid memory overhead of torchdiffeq and to stay on the gpu,
    # ! since (while scipy is less memory intensive) it is cpu native and therefore lacks the performance
    # ! enhancements of the gpu arch

    def kac_explicit_euler(self, model: torch.nn.Module, x_batch: torch.Tensor, 
                           t_start: torch.Tensor, dt: float, num_substeps: int) -> torch.Tensor:
        """Explicit Euler stepping for Kac models over [t, t - dt] in
        num_substeps many steps."""
        dt_sub = dt / num_substeps
        x_curr = x_batch.clone()
        t_curr = t_start.clone()

        with torch.no_grad():
            for _ in range(num_substeps):
                v = model(x_curr, t_curr)
                x_curr = x_curr - v * dt_sub
                t_curr = t_curr - dt_sub
                
        return x_curr

    def kac_rk2(self, model: torch.nn.Module, x_batch: torch.Tensor, 
                t_start: torch.Tensor, dt: float, num_substeps: int) -> torch.Tensor:
        """RK2 (Midpoint) stepping for Kac models over [t, t - dt] in
        num_substeps many steps."""
        dt_sub = dt / num_substeps
        x_curr = x_batch.clone()
        t_curr = t_start.clone()

        with torch.no_grad():
            for _ in range(num_substeps):
                # Step 1: Evaluate at t
                v1 = model(x_curr, t_curr)
                
                # Step 2: Evaluate at midpoint
                x_mid = x_curr - v1 * (dt_sub / 2)
                t_mid = t_curr - (dt_sub / 2)
                v2 = model(x_mid, t_mid)
                
                # Full step using midpoint velocity
                x_curr = x_curr - v2 * dt_sub
                t_curr = t_curr - dt_sub
                
        return x_curr


    # =============================================================================================
    # diffusion reverse time sde solver with euler maruyama scheme


    def diffusion_euler_maruyama(self, model: torch.nn.Module, x_batch: torch.Tensor,
                                 t_start: torch.Tensor, dt: torch.Tensor = torch.ones(1) * -1,
                                 num_substeps: int = 1, noise_injection_bool: bool = True) -> torch.Tensor:
        """Integrates the teacher over [t*, t* - delta_t] using num_substeps many uniform substeps

        This is the Euler-Maruyama Scheme also used to solve the SDE formulation of the reverse process.
        """
        # * Euler Maruyama Scheme as described in "Song et al 2021"
        from Cluster.utils.diffusion import f, g, b

        # Safeguard: Vectorized clipping of dt to ensure t_start - dt >= time_truncation
        if not isinstance(dt, torch.Tensor):
            dt = torch.full_like(t_start, dt)
        elif dt.numel() == 1:
            dt = dt.expand_as(t_start)
            
        max_allowed_dt = torch.clamp(t_start - self.args.time_truncation, min=0.0)
        dt = torch.minimum(dt, max_allowed_dt)

        dt_sub = dt / num_substeps
        x_curr = x_batch.clone()
        t_curr = t_start.clone()
        
        # reverse the batch on [t*, t* - dt]
        with torch.no_grad():
            for _ in range(num_substeps):

                # Get continuous coefficients
                f_t_x = f(t_curr, x_curr)
                g_t = g(t_curr).view(-1, 1, 1, 1)
                b_t = b(t_curr).view(-1, 1, 1, 1)

                # Predict score using continuous time
                pred_noise = model(x_curr, t_curr)

                # Need to clamp the variance to prevent a division by zero which will throw NaNs in pytorch
                # Because otherwise for small values of t the difference between 1 and b_t falls below the
                # machine epsilon and is thus evaluated as 0
                variance = torch.clamp(1 - b_t**2, min=1e-8)
                pred_score = - pred_noise / torch.sqrt(variance)

                # Scale updates explicitly by dt and sqrt(dt)
                drift_update = f_t_x * dt_sub.view(-1, 1, 1, 1)
                score_update = (g_t ** 2) * pred_score * dt_sub.view(-1, 1, 1, 1)
                if noise_injection_bool:
                    noise_injection = g_t * torch.sqrt(dt_sub).view(-1, 1, 1, 1) * torch.randn_like(x_curr)
                else:
                    noise_injection = 0.0

                # Continuous SDE reverse step formula
                x_curr = x_curr - drift_update + score_update + noise_injection
                t_curr -= dt_sub

        return x_curr
    

    # =============================================================================================
    # ODE derivative functions


    class DiffusionODEDerivative(torch.nn.Module):
        """PyTorch derivative wrapper for the probability flow ODE."""
        def __init__(self, model: torch.nn.Module, min_t: float):
            super().__init__()
            self.model = model
            self.nfe = 0
            self.min = min_t

        def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            from Cluster.utils.diffusion import f, g, b

            self.nfe += 1

            # Safeguard: prevent adaptive solver overshoot into negative time
            t_safe = torch.clamp(t, min=self.min)

            # Broadcast scalar t to batch size
            t_vec = torch.full((x.shape[0],), float(t_safe), device=x.device)
            
            f_t_x = f(t_vec, x)
            g_t = g(t_vec).view(-1, 1, 1, 1)
            b_t = b(t_vec).view(-1, 1, 1, 1)
            
            pred_noise = self.model(x, t_vec)

            # Numerical safeguard: prevent division by zero or sqrt of negative numbers
            variance = torch.clamp(1 - b_t**2, min=1e-8)
            pred_score = -pred_noise / torch.sqrt(variance)
            
            # * Diffusion Inference using Probability Flow ODE as described in Song et al. 2021
            # Probability flow ODE: dx/dt = f(x,t) - 0.5 * g(t)^2 * score
            dx_dt = f_t_x - 0.5 * (g_t ** 2) * pred_score

            # Sanitize output to guarantee the adaptive solver doesn't underflow on anomalies
            return torch.nan_to_num(dx_dt, nan=0.0, posinf=1e5, neginf=-1e5)


    class KacODEDerivative(torch.nn.Module):
        """
        A clean, localized PyTorch derivative wrapper matching torchdiffeq's 
        expected signature: fn(t, x) -> dx/dt
        """
        def __init__(self, model: torch.nn.Module, channels: int, width: int, height: int):
            super().__init__()
            self.model = model
            self.channels = channels
            self.width = width
            self.height = height
            self.nfe = 0

        def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

            self.nfe += 1

            B = x.shape[0]
            # Reshape 1D flattened vectors back to image tensors for the UNet
            x_img = x.view(B, self.channels, self.width, self.height)
            
            # Broadcast the scalar t to match batch size
            t_vec = torch.full((B,), float(t), device=x.device)
            
            # Predict the velocity field
            v = self.model(x_img, t_vec)
                
            # Flatten back to match the solver's state tracking configuration
            return v.view(x.shape)