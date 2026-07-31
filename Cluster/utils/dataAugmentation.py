import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KarrasAugmentationPipeline(nn.Module):
    """This implements the GAN style geometric transformations describe in
    2022 - Karras et al - Elucidating the design space of diffusion based generative models"""

    def __init__(self, p: float = 0.12):
        super().__init__()

        self.p = p

    def forward(self, x: torch.Tensor):

        B = x.shape[0]
        device = x.device
        
        # Base identity matrix [B, 3, 3]
        I = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
        
        # X-Flip (100% prob)
        a0 = torch.randint(0, 2, size=(B,), device=device, dtype=torch.float32)
        M_xflip = I.clone()
        M_xflip[:, 0, 0] = 1 - 2 * a0
        
        # Y-Flip (Prob = p)
        m1 = (torch.rand(B, device=device) < self.p).float()
        a1 = torch.randint(0, 2, size=(B,), device=device, dtype=torch.float32) * m1
        M_yflip = I.clone()
        M_yflip[:, 1, 1] = 1 - 2 * a1
        
        # Scaling (Prob = p)
        m2 = (torch.rand(B, device=device) < self.p).float()
        a2 = torch.randn(B, device=device) * m2
        s = (2 ** 0.2) ** a2
        M_scale = I.clone()
        M_scale[:, 0, 0] = s
        M_scale[:, 1, 1] = s
        
        # Rotation (Prob = p)
        m3 = (torch.rand(B, device=device) < self.p).float()
        a3 = (torch.rand(B, device=device) * 2 * math.pi - math.pi) * m3
        M_rot = I.clone()
        M_rot[:, 0, 0] = torch.cos(-a3)
        M_rot[:, 0, 1] = -torch.sin(-a3)
        M_rot[:, 1, 0] = torch.sin(-a3)
        M_rot[:, 1, 1] = torch.cos(-a3)
        
        # Anisotropy (Prob = p)
        m45 = (torch.rand(B, device=device) < self.p).float()
        a4 = (torch.rand(B, device=device) * 2 * math.pi - math.pi) * m45
        a5 = torch.randn(B, device=device) * m45
        s_aniso = (2 ** 0.2) ** a5
        
        M_rot_aniso = I.clone()
        M_rot_aniso[:, 0, 0] = torch.cos(a4)
        M_rot_aniso[:, 0, 1] = -torch.sin(a4)
        M_rot_aniso[:, 1, 0] = torch.sin(a4)
        M_rot_aniso[:, 1, 1] = torch.cos(a4)
        
        M_scale_aniso = I.clone()
        M_scale_aniso[:, 0, 0] = s_aniso
        M_scale_aniso[:, 1, 1] = 1 / s_aniso
        
        M_rot_aniso_inv = I.clone()
        M_rot_aniso_inv[:, 0, 0] = torch.cos(-a4)
        M_rot_aniso_inv[:, 0, 1] = -torch.sin(-a4)
        M_rot_aniso_inv[:, 1, 0] = torch.sin(-a4)
        M_rot_aniso_inv[:, 1, 1] = torch.cos(-a4)
        
        M_aniso = M_rot_aniso @ M_scale_aniso @ M_rot_aniso_inv
        
        # 6. Translation (Prob = p)
        m67 = (torch.rand(B, device=device) < self.p).float()
        a6 = torch.randn(B, device=device) * m67
        a7 = torch.randn(B, device=device) * m67
        M_trans = I.clone()
        M_trans[:, 0, 2] = a6 * (1/8)
        M_trans[:, 1, 2] = a7 * (1/8)
        
        # Combine transformations
        M = M_trans @ M_aniso @ M_rot @ M_scale @ M_yflip @ M_xflip
        
        # Apply combined affine transformation
        M_inv = torch.inverse(M)[:, :2, :]
        grid = F.affine_grid(M_inv, x.size(), align_corners=False)
        x_aug = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        
        # Construct 9-dimensional conditioning vector
        cond = torch.stack([
            a0,
            a1,
            a2,
            torch.cos(a3) - 1,
            torch.sin(a3),
            a5 * torch.cos(a4),
            a5 * torch.sin(a4),
            a6,
            a7
        ], dim=1)
        
        return x_aug, cond