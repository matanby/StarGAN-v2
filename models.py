from typing import Optional, Callable

import torch
from torch import nn

import utils


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

    def forward(self,
                z: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_tensor(z, num_dims=2)
        assert utils.is_valid_tensor(y, num_dims=2, batch_size=z.shape[0])

        x = z
        for layer in self._body:
            x = layer(x)
            x = nn.functional.relu(x)

        style_code = [layer(x) for layer in self._heads]
        style_code = torch.stack(style_code, dim=1)
        style_code = style_code[:, y]

        return style_code


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

        assert utils.is_valid_image_tensor(x)
        assert utils.is_valid_tensor(style_code, num_dims=2, batch_size=x.shape[0])

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

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_image_tensor(x)
        assert utils.is_valid_tensor(y, num_dims=2, batch_size=x.shape[0])

        for block in self._down_blocks:
            x = block(x)

        outputs = [layer(x) for layer in self._heads]
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs[:, y]
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

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_image_tensor(x)
        assert utils.is_valid_tensor(y, num_dims=2, batch_size=x.shape[0])

        for block in self._down_blocks:
            x = block(x)

        outputs = [layer(x) for layer in self._heads]
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs[:, y]

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

        # TODO: is this right?
        if in_channels == out_channels:
            self._skip_fn = lambda x: x
        else:
            self._skip_fn = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)

        # TODO: how many conv layers in each block?
        self._norm_1 = self._create_norm_layer(norm, in_channels, style_code_dim)
        self._conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self._norm_2 = self._create_norm_layer(norm, out_channels, style_code_dim)
        self._conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)

        self._resample = self._create_resample_layer(resample)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        # TODO: when to normalize?
        # TODO: when to resample?

        if isinstance(self._resample, nn.UpsamplingNearest2d):
            x = self._resample(x)

        x_skip = self._skip_fn(x)

        # norm -> activation -> conv
        x = self._norm_1(x, style_code)
        x = nn.functional.relu(x)
        x = self._conv_1(x)

        # norm -> activation -> conv
        x = self._norm_2(x, style_code)
        x = nn.functional.relu(x)
        x = self._conv_2(x)

        x = x + x_skip

        if isinstance(self._resample, nn.AvgPool2d):
            x = self._resample(x)

        return x

    @staticmethod
    def _create_norm_layer(norm: str,
                           num_channels: int,
                           style_code_dim: int) -> Callable:
        norm = norm.lower()
        if norm is None:
            return lambda x: x
        if norm == 'in':
            # TODO: use/not learned affine transformation?
            norm_layer = nn.InstanceNorm2d(num_channels)
            return lambda x, style: norm_layer(x)
        elif norm == 'adain':
            norm_layer = AdaptiveInstanceNorm2d(num_channels, style_code_dim)
            return norm_layer
        else:
            raise ValueError(f'Invalid normalization method: "{norm}"')

    @staticmethod
    def _create_resample_layer(resample: str) -> Optional[Callable]:
        resample = resample.lower()
        if resample is None:
            return None
        elif resample == 'avg_pool':
            return nn.AvgPool2d(kernel_size=2)
        elif resample == 'nn':
            return nn.UpsamplingNearest2d(scale_factor=2)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), -1)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self,
                 num_channels: int,
                 style_code_dim: int):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self._num_channels = num_channels
        self._instance_norm = nn.InstanceNorm2d(num_channels)
        # TODO: single mu/sigma for all channels / unique pair for each channel?
        self._linear = nn.Linear(style_code_dim, num_channels * 2)

    def forward(self,
                x: torch.Tensor,
                style_code: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_tensor(x, num_dims=4)
        assert x.shape[1] == self._num_channels

        x = self._instance_norm(x)
        stats = self._linear(style_code)
        mu = stats[:, :self._num_channels]
        sigma = stats[:, self._num_channels:]
        x = x * sigma.expand_as(x) + mu.expand_as(x)
        return x
