from typing import List, Optional

import torch
from torch import nn


class Mapping(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 num_heads: int):

        super().__init__()
        self._body = [nn.Linear(latent_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            self._body.append(nn.Linear(hidden_dim, hidden_dim))

        self._heads = [nn.Linear(hidden_dim, out_dim) for _ in range(num_heads)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        for layer in self._body:
            x = layer(x)
            x = nn.functional.relu(x)

        outputs = [layer(x) for layer in self._heads]
        return outputs


class Generator(nn.Module):
    def __init__(self, style_code_dim: int):
        super(Generator, self).__init__()
        self._down_blocks = [
            nn.Conv2d(3, 32, kernel_size=1, padding=1),
            ResBlock(32, 64, kernel_size=3, resample='avg_pool', norm='IN'),
            ResBlock(64, 128, kernel_size=3, resample='avg_pool', norm='IN'),
            ResBlock(128, 256, kernel_size=3, resample='avg_pool', norm='IN'),
            ResBlock(256, 512, kernel_size=3, resample='avg_pool', norm='IN'),
        ]

        self._inter_blocks = [
            ResBlock(512, 512, kernel_size=3, resample=None, norm='IN'),
            ResBlock(512, 512, kernel_size=3, resample=None, norm='IN'),
            ResBlock(512, 512, kernel_size=3, resample=None, norm='AdaIN', style_code_dim=style_code_dim),
            ResBlock(512, 512, kernel_size=3, resample=None, norm='AdaIN', style_code_dim=style_code_dim),
        ]

        self._up_blocks = [
            ResBlock(512, 256, kernel_size=3, resample='NN', norm='AdaIN', style_code_dim=style_code_dim),
            ResBlock(256, 128, kernel_size=3, resample='NN', norm='AdaIN', style_code_dim=style_code_dim),
            ResBlock(128, 64, kernel_size=3, resample='NN', norm='AdaIN', style_code_dim=style_code_dim),
            ResBlock(64, 32, kernel_size=3, resample='NN', norm='AdaIN', style_code_dim=style_code_dim),
            nn.Conv2d(32, 3, kernel_size=1, padding=1),
        ]

        self._all_blocks = self._down_blocks + self._inter_blocks + self._up_blocks

    def forward(self,
                x: torch.Tensor,
                style_code: torch.Tensor) -> torch.Tensor:

        for block in self._down_blocks:
            x = block(x, style_code)

        for block in self._inter_blocks:
            x = block(x, style_code)

        for block in self._up_blocks:
            x = block(x, style_code)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_heads: int):
        super(Discriminator, self).__init__()
        self._down_blocks = [
            nn.Conv2d(3, 32, kernel_size=1, padding=1),
            ResBlock(32, 64, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(64, 128, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(128, 256, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(256, 512, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(512, 1024, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(1024, 1024, kernel_size=3, resample='avg_pool', norm=None),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, kernel_size=4, padding=1),
            nn.LeakyReLU(0.2),
            Flatten()
        ]

        self._heads = [nn.Linear(1024, 1) for _ in range(num_heads)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        for block in self._down_blocks:
            x = block(x)

        outputs = [layer(x) for layer in self._heads]
        return outputs


class StyleEncoder(nn.Module):
    def __init__(self,
                 style_code_dim: int,
                 num_heads: int):

        super(StyleEncoder, self).__init__()
        self._down_blocks = [
            nn.Conv2d(3, 16, kernel_size=1, padding=1),
            ResBlock(16, 32, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(32, 64, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(64, 128, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(128, 256, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(256, 512, kernel_size=3, resample='avg_pool', norm=None),
            ResBlock(512, 512, kernel_size=3, resample='avg_pool', norm=None),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, padding=1),
            nn.LeakyReLU(0.2),
            Flatten()
        ]

        self._heads = [nn.Linear(512, style_code_dim) for _ in range(num_heads)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        for block in self._down_blocks:
            x = block(x)

        outputs = [layer(x) for layer in self._heads]
        return outputs


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 resample: Optional[str] = None,
                 norm: Optional[str] = None,
                 style_code_dim: Optional[int] = None):
        super().__init__()

        # TODO: how many conv layers in each block?
        self._conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self._conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        # TODO: is this right?
        self._conv_resid = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
        self._create_norm_layer(norm, out_channels, style_code_dim)
        self._create_resample_layer(resample)

    def _create_norm_layer(self,
                           norm: str,
                           out_channels: int,
                           style_code_dim: int) -> None:
        norm = norm.lower()
        if norm is None:
            self._norm = lambda x: x
        if norm == 'in':
            # TODO: use/not learned affine transformation?
            self._norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'adain':
            self._norm = AdaptiveInstanceNorm2d(out_channels, style_code_dim)
        else:
            raise ValueError(f'Invalid normalization method: "{norm}"')

    def _create_resample_layer(self, resample: str):
        resample = resample.lower()
        if resample is None:
            self._resample = lambda x: x
        elif resample == 'avg_pool':
            self._resample = nn.AvgPool2d(kernel_size=2)
        elif resample == 'nn':
            self._resample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        x_skip = self._conv_resid(x)
        x = self._conv_1(x)
        x = nn.functional.relu(x)
        x = self._conv_2(x)
        x = x + x_skip
        x = nn.functional.relu(x)
        # TODO: when to normalize?
        x = self._norm(x, style_code)
        # TODO: when to resample?
        x = self._resample(x)
        return x


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self,
                 num_channels: int,
                 style_code_dim: int):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self._instance_norm = nn.InstanceNorm2d(num_channels)
        # TODO: single mu/sigma for all channels / unique pair for each channel?
        self._linear = nn.Linear(style_code_dim, 2)

    def forward(self,
                x: torch.Tensor,
                style_code: torch.Tensor) -> torch.Tensor:

        x = self._instance_norm(x)
        stats = self._linear(style_code)
        mu = stats[0]
        sigma = stats[1]
        x = x * sigma.expand_as(x) + mu.expand_as(x)
        return x
