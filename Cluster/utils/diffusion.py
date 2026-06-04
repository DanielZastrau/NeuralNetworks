import torch

class Diffusion:

    @staticmethod
    def beta(t: torch.Tensor) -> torch.Tensor:
        """In Sohl-Dickstein 2015 they use a linear schedule with a step of 1 / T"""

        t = torch.as_tensor(t, dtype=torch.float32, device=t.device)
        beta_start = torch.tensor(0.01, dtype=t.dtype, device=t.device)
        beta_end = torch.tensor(20.0, dtype=t.dtype, device=t.device)

        # linear interpolation between beta_start and beta_end
        return beta_start + t * (beta_end - beta_start)

    @staticmethod
    def f(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return -0.5 * Diffusion.beta(t).view(-1, 1, 1, 1) * x_t

    @staticmethod
    def g(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(Diffusion.beta(t))

    @staticmethod
    def h(t: torch.Tensor) -> torch.Tensor:
        """integral from 0 to t of beta(s) ds, which is used in the closed-form solution of the SDE
        Using Sohl-Dickstein 2015's linear schedule, we can compute this integral in closed form as:
        h(t) = (beta_start * t + 0.5 * (beta_end - beta_start) * t^2) / T
        """
        beta_start = torch.tensor(0.01, dtype=t.dtype, device=t.device)
        beta_end = torch.tensor(20.0, dtype=t.dtype, device=t.device)

        # Sohl-Dickstein 2015's linear schedule with a step of 1 / T
        return torch.as_tensor(t*(beta_start + (t / 2) * (beta_end - beta_start)), dtype=torch.float32, device=t.device)

    @staticmethod
    def b(t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-Diffusion.h(t) / 2)

    @staticmethod
    def velocity(t: torch.Tensor, x: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        
        f_t_x = Diffusion.f(t, x)
        g_t = Diffusion.g(t).view(-1, 1, 1, 1)
        b_t = Diffusion.b(t).view(-1, 1, 1, 1)
        
        pred_noise = model(x, t)

        # Numerical safeguard: prevent division by zero or sqrt of negative numbers
        variance = torch.clamp(1 - b_t**2, min=1e-8)
        pred_score = -pred_noise / torch.sqrt(variance)
        
        # * Diffusion Inference using Probability Flow ODE as described in Song et al. 2021
        # Probability flow ODE: dx/dt = f(x,t) - 0.5 * g(t)^2 * score
        return f_t_x - 0.5 * (g_t ** 2) * pred_score