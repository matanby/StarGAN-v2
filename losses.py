import torch
from torch import nn

import utils
from models import Generator, Discriminator, StyleEncoder


class AdversarialLoss(nn.Module):
    def __init__(self, d: Discriminator):
        super().__init__()
        self._d = d

    def forward(self,
                x_real: torch.Tensor,
                y_real: torch.Tensor,
                x_fake: torch.Tensor,
                y_fake: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_image_tensor(x_real)
        assert utils.is_valid_tensor(y_real, num_dims=2, batch_size=x_real.shape[0])
        assert utils.is_valid_image_tensor(x_fake)
        assert utils.is_valid_tensor(y_fake, num_dims=2, batch_size=x_fake.shape[0])

        d_real = self._d(x_real, y_real)
        d_fake = self._d(x_fake, y_fake)

        l_adv = (
            torch.mean(torch.log(d_real), dim=0) +
            torch.mean(torch.log(1 - d_fake), dim=0)
        )

        return l_adv


class StyleReconstructionLoss(nn.Module):
    def __init__(self, e: StyleEncoder):
        super().__init__()
        self._e = e
        self._loss_fn = torch.nn.L1Loss()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                s_x: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_image_tensor(x)
        assert utils.is_valid_tensor(y, num_dims=2, batch_size=x.shape[0])
        assert utils.is_valid_tensor(s_x, num_dims=2, batch_size=x.shape[0])

        e_x = self._e(x, y)
        l_sty = self._loss_fn(e_x, s_x)
        return l_sty


class StyleDiversificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss_fn = torch.nn.L1Loss()

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_image_tensor(x1)
        assert utils.is_valid_image_tensor(x2)
        assert x1.shape[0] == x2.shape[0]

        l_ds = self._loss_fn(x1, x2)
        return l_ds


class CycleConsistencyLoss(nn.Module):
    def __init__(self, e: StyleEncoder, g: Generator):
        super().__init__()
        self._e = e
        self._g = g
        self._loss_fn = nn.L1Loss()

    def forward(self,
                x_real: torch.Tensor,
                y_real: torch.Tensor,
                x_fake: torch.Tensor) -> torch.Tensor:

        assert utils.is_valid_image_tensor(x_real)
        assert utils.is_valid_image_tensor(x_fake)
        assert x_real.shape[0] == x_fake.shape[0]
        assert utils.is_valid_tensor(y_real, num_dims=2, batch_size=x_real.shape[0])

        s_x = self._e(x_real, y_real)
        x_recon = self._g(x_fake, s_x)
        l_cyc = self._loss_fn(x_real, x_recon)
        return l_cyc