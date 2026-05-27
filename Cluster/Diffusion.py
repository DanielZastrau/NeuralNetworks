import torch

# positive increasing time schedule
def beta(t: torch.Tensor) -> torch.Tensor:
    """In Sohl-Dickstein 2015 they use a linear schedule with a step of 1 / T"""

    t = torch.as_tensor(t, dtype=torch.float32, device=t.device)
    beta_start = torch.tensor(0.01, dtype=t.dtype, device=t.device)
    beta_end = torch.tensor(20.0, dtype=t.dtype, device=t.device)

    # linear interpolation between beta_start and beta_end
    return beta_start + t * (beta_end - beta_start)

    # # Sohl-Dickstein 2015's linear schedule with a step of 1 / T
    # return (beta_start + (t / T) * (beta_end - beta_start))

def f(t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    t = t.view(-1, 1, 1, 1)  # Reshape t to (B, 1, 1, 1) for broadcasting

    return -0.5 * beta(t) * x_t

def g(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(beta(t))

def h(t: torch.Tensor) -> torch.Tensor:
    """integral from 0 to t of beta(s) ds, which is used in the closed-form solution of the SDE
    Using Sohl-Dickstein 2015's linear schedule, we can compute this integral in closed form as:
    h(t) = (beta_start * t + 0.5 * (beta_end - beta_start) * t^2) / T
    """
    beta_start = torch.tensor(0.01, dtype=t.dtype, device=t.device)
    beta_end = torch.tensor(20.0, dtype=t.dtype, device=t.device)

    # Sohl-Dickstein 2015's linear schedule with a step of 1 / T
    return torch.as_tensor(t*(beta_start + (t / 2) * (beta_end - beta_start)), dtype=torch.float32, device=t.device)

def b(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-h(t) / 2)