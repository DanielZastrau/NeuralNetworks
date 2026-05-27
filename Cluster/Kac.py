import torch

def f(t: torch.Tensor, T: float, name: str = 'linear') -> torch.Tensor:
    if name == 'linear':
        return 1 - t / T
    elif name == 'exp':
        return torch.exp(-t)
    else:
        raise ValueError(f"Unknown schedule: {name}")

def df(t: torch.Tensor, T: float, name: str = 'linear') -> torch.Tensor:
    if name == 'linear':
        return -1.0 / T * torch.ones_like(t)
    elif name == 'exp':
        return -torch.exp(-t)
    else:
        raise ValueError(f"Unknown schedule: {name}")

def g(t: torch.Tensor, T: float, name: str = 't2') -> torch.Tensor:
    """Computes the time reparameterization g(t) for the noise process."""
    if name == 't':
        return t
    elif name == 't2':
        # Normalized such that g(T) = T
        return T * (t / T)**2
    else:
        raise ValueError(f"Unknown g schedule: {name}")

def dg(t: torch.Tensor, T: float, name: str = 't2') -> torch.Tensor:
    """Computes the time reparameterization g(t) for the noise process."""
    if name == 't':
        return 1
    elif name == 't2':
        # Normalized such that g(T) = T
        return 2*t/T
    else:
        raise ValueError(f"Unknown g schedule: {name}")