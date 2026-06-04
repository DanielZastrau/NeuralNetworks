import torch

class MMD:
    # * the scheduling functions f, g as proposed in 2025 - Duong Chemseddine Kornhardt - Adapting Noise to Data

    @staticmethod
    def f(t: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return 1 - t / T
    
    @staticmethod
    def df(t: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return torch.ones_like(t, device=t.device) * -1 / T
    
    @staticmethod
    def g(t: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return t / T
    
    @staticmethod
    def dg(t: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        return torch.ones_like(t, device=t.device) / T
    
    @staticmethod
    def get_noise(t: torch.Tensor, x:torch.Tensor, b: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Because of the target simplification in the loss fns module I also return the pre_noise
        Because of its use in the sampling, noisifier and loss fns modules I also return the full noise
        """

        # t should have shape [B], x should have shape [B, data_dim.channels, data_dim.width, data_dim.height]

        # sample uniform noise and normalize to [-1, 1],    pre_noise should be of the same shape as x
        pre_noise = torch.rand_like(x, device=x.device) * 2 - 1

        # transform it according to the noise process
        return pre_noise, b * (1 - torch.exp(- t / b)).view(-1, 1, 1, 1) * pre_noise
