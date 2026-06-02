import torch

class MMD:
    # * the scheduling functions f, g as proposed in 2025 - Duong Chemseddine Kornhardt - Adapting Noise to Data
    # TODO should normalize them at some point to the case where we train over [0, T] instead of [0, 1]

    @staticmethod
    def f(t: torch.Tensor) -> torch.Tensor:
        return 1 - t
    
    @staticmethod
    def df(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t, device=t.device) * -1
    
    @staticmethod
    def g(t: torch.Tensor) -> torch.Tensor:
        return t
    
    @staticmethod
    def dg(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t, device=t.device)