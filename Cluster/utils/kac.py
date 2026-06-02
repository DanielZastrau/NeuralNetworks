import torch

class Kac:

    @staticmethod
    def f(t: torch.Tensor, T: float, name: str) -> torch.Tensor:
        # if name == 'opt1':
        return 1 - t / T

    @staticmethod
    def df(t: torch.Tensor, T: float, name: str) -> torch.Tensor:
        # if name == 'opt1':
        return -1.0 / T * torch.ones_like(t)

    @staticmethod
    def g(t: torch.Tensor, T: float, name: str) -> torch.Tensor:
        """Computes the time reparameterization g(t) for the noise process."""
        if name == 'opt1':
            return t
        else:    # name == 'opt2':
            # Normalized such that g(T) = T
            return T * (t / T)**2

    @staticmethod
    def dg(t: torch.Tensor, T: float, name: str) -> torch.Tensor:
        """Computes the time reparameterization g(t) for the noise process."""
        if name == 'opt1':
            return torch.ones_like(t, device=t.device)
        else:    # name == 'opt2':
            return 2*t/T