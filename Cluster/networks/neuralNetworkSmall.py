import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """Maps continuous time steps to positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
    
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        
        half_dim = self.dim // 2
        
        embeddings = math.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        

        if self.dim % 2:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

class TimeAwareDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => Dropout) * 2 with Time Embedding Injection"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout_prob: float = 0.2):
        super().__init__()
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )

        # Projects the shared time embedding to the current layer's channel dimension
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        
        # Project time and broadcast spatially (B, C) -> (B, C, 1, 1)
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Inject time conditioning via addition
        x = x + time_emb 
        
        x = self.conv2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then time-aware double conv"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.conv = TimeAwareDoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.conv(self.maxpool(x), t_emb)


class Up(nn.Module):
    """Upscaling then time-aware double conv"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = TimeAwareDoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x, t_emb)

class OutConv(nn.Module):
    """Final 1x1 convolution to map to desired number of classes"""
    def __init__(self, in_channels: int, out_channels: int):

        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ConditionalUNet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        # Time Embedding MLP (Shared across all layers)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        dropout_prob = 0.2

        # Encoder
        self.inc = TimeAwareDoubleConv(self.in_channels, 64, time_emb_dim, dropout_prob=dropout_prob)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 512, time_emb_dim)
        self.down4 = Down(512, 1024, time_emb_dim) 

        # Decoder
        self.up1 = Up(1024, 512, time_emb_dim)
        self.up2 = Up(512, 256, time_emb_dim)
        self.up3 = Up(256, 128, time_emb_dim)
        self.up4 = Up(128, 64, time_emb_dim)
        
        self.outc = OutConv(64, self.out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1. Generate base time embedding
        t_emb = self.time_mlp(t)

        # 2. Forward pass with time injection at every block
        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb) 
        
        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        
        logits = self.outc(x)
        return logits