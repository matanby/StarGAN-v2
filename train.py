import os
import sys

import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import configs
import data
import losses
import models


class Trainer:
    def __init__(self, config_name: str):
        self._curr_iter = 0
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_config(config_name)
        self._create_data_loader()
        self._create_models()
        self._create_losses()
        self._create_optimizers()
        self._create_outputs_dir()
        self._create_tensorboard_writer()

    def train(self):
        cfg = self._config
        device = self._device
        generator = self._generator
        mapping = self._mapping
        mapping_ema = self._mapping_ema
        style_encoder = self._style_encoder
        style_encoder_ema = self._style_encoder_ema
        discriminator = self._discriminator
        adv_loss = self._adv_loss
        style_recon_loss = self._style_recon_loss
        style_diverse_loss = self._style_diverse_loss
        cycle_loss = self._cycle_loss
        opt_generator = self._opt_generator
        opt_mapping = self._opt_mapping
        opt_style_encoder = self._opt_style_encoder
        opt_discriminator = self._opt_discriminator
        curr_iter = self._curr_iter

        models_save_path = os.path.join(self._model_dir, 'models.pt')
        if os.path.exists(models_save_path):
            print('Loading previously saved models from PT file...')
            self._load_models(models_save_path)

        prog_bar = tqdm(total=cfg.training_iterations)
        while True:
            for batch in self._data_loader:
                if curr_iter >= cfg.training_iterations:
                    break

                x_real = batch['image'].to(device)
                y_real = batch['attributes']['gender'].to(device).reshape(-1, 1)

                z = torch.randn((cfg.batch_size, cfg.mapper_latent_code_dim)).to(device)
                z_2 = torch.randn((cfg.batch_size, cfg.mapper_latent_code_dim)).to(device)
                y_fake = torch.randint(cfg.num_domains, size=(cfg.batch_size, 1)).to(device)
                s = mapping(z, y_fake)
                s_2 = mapping(z_2, y_fake)
                x_fake = generator(x_real, s)
                x_fake_2 = generator(x_real, s_2)

                l_adv_d, l_adv_g, r1_reg = adv_loss(x_real, y_real, x_fake, y_fake)
                l_sty = style_recon_loss(x_fake, y_fake, s)
                l_ds = style_diverse_loss(x_fake, x_fake_2)
                l_cyc = cycle_loss(x_real, y_real, x_fake)

                l_d = l_adv_d + cfg.lambda_r1 * r1_reg
                lambda_ds = max(0.0, cfg.lambda_ds * (1 - curr_iter / cfg.ds_loss_iterations))
                l_fge = (
                    l_adv_g
                    + cfg.lambda_sty * l_sty
                    - lambda_ds * l_ds
                    + cfg.lambda_cyc * l_cyc
                )

                generator.zero_grad()
                mapping.zero_grad()
                style_encoder.zero_grad()

                l_fge.backward(retain_graph=True)
                opt_generator.step()
                opt_mapping.step()
                opt_style_encoder.step()

                discriminator.zero_grad()
                l_d.backward()
                opt_discriminator.step()

                self._update_ema_models()

                l_adv_d_np = l_adv_d.to("cpu").detach().numpy()
                l_adv_g_np = l_adv_g.to("cpu").detach().numpy()
                r1_reg_np = r1_reg.to("cpu").detach().numpy()
                l_sty_np = l_sty.to("cpu").detach().numpy()
                l_ds_np = l_ds.to("cpu").detach().numpy()
                l_cyc_np = l_cyc.to("cpu").detach().numpy()
                l_fge_np = l_fge.to("cpu").detach().numpy()
                l_d_np = l_d.to("cpu").detach().numpy()
                status = (
                    f'L_adv_D: {l_adv_d_np:.2f}, '
                    f'L_adv_G: {l_adv_g_np:.2f}, '
                    f'R1_reg: {r1_reg_np:.2f}, '
                    f'L_sty: {l_sty_np:.2f}, '
                    f'L_ds: {l_ds_np:.2f}, '
                    f'L_cyc: {l_cyc_np:.2f}, '
                    f'L_D: {l_d_np:.2f}, '
                    f'L_FGE: {l_fge_np:.2f}'
                )
                prog_bar.set_description(status)

                is_last_iter = curr_iter == cfg.training_iterations - 1
                if (curr_iter + 1) % cfg.tb_losses_log_interval == 0 or is_last_iter:
                    self._summary_writer.add_scalar('loss/L_adv_D', l_adv_d, curr_iter)
                    self._summary_writer.add_scalar('loss/L_adv_G', l_adv_g, curr_iter)
                    self._summary_writer.add_scalar('loss/R1_reg', r1_reg, curr_iter)
                    self._summary_writer.add_scalar('loss/L_sty', l_sty, curr_iter)
                    self._summary_writer.add_scalar('loss/L_ds', l_ds, curr_iter)
                    self._summary_writer.add_scalar('loss/L_cyc', l_cyc, curr_iter)
                    self._summary_writer.add_scalar('loss/L_D', l_d, curr_iter)
                    self._summary_writer.add_scalar('loss/L_FGE', l_fge, curr_iter)
                    self._summary_writer.add_scalar('loss/L_adv_D', l_adv_d, curr_iter)

                if (curr_iter + 1) % cfg.tb_samples_log_interval == 0 or is_last_iter:
                    prog_bar.set_description('Saving sample images...')
                    sample_images = torch.cat((x_real, x_fake))
                    sample_save_path = os.path.join(self._samples_dir, f'iter_{curr_iter}.jpg')
                    torchvision.utils.save_image(
                        tensor=sample_images,
                        filename=sample_save_path,
                        nrow=cfg.batch_size,
                        normalize=True,
                        range=(-1, 1)
                    )

                    with torch.no_grad():
                        s_ema = mapping_ema(z, y_fake)
                        x_fake_ema = self._generator_ema(x_real, s_ema)
                        s_x_ema = style_encoder_ema(x_real, y_real)
                        x_fake_recon_ema = self._generator_ema(x_real, s_x_ema)
                    sample_images_ema = torch.cat((x_real, x_fake_ema))
                    sample_save_path = os.path.join(self._samples_dir, f'iter_{curr_iter}_ema.jpg')
                    torchvision.utils.save_image(
                        tensor=sample_images_ema,
                        filename=sample_save_path,
                        nrow=cfg.batch_size,
                        normalize=True,
                        range=(-1, 1)
                    )

                    x_real_clamp = ((x_real + 1) / 2).clamp(0, 1)
                    x_fake_clamp = ((x_fake + 1) / 2).clamp(0, 1)
                    x_fake_ema_clamp = ((x_fake_ema + 1) / 2).clamp(0, 1)
                    x_fake_recon_ema_clamp = ((x_fake_recon_ema + 1) / 2).clamp(0, 1)
                    self._summary_writer.add_images('samples/real', x_real_clamp, curr_iter)
                    self._summary_writer.add_images('samples/generated', x_fake_clamp, curr_iter)
                    self._summary_writer.add_images('samples/generated_ema', x_fake_ema_clamp, curr_iter)
                    self._summary_writer.add_images('samples/recon_ema', x_fake_recon_ema_clamp, curr_iter)

                if (curr_iter + 1) % cfg.model_snapshot_interval == 0 or is_last_iter:
                    prog_bar.set_description('Saving models...')
                    self._curr_iter = curr_iter
                    self._save_models(models_save_path)

                curr_iter += 1
                prog_bar.update(1)

    def _load_config(self, config_name: str) -> None:
        config: configs.TrainConfig = getattr(configs, config_name, None)
        if config is None:
            raise ValueError(f'configuration "{config_name}" not found in {configs.__file__}')

        print(config.str())
        self._config = config

    def _create_data_loader(self) -> None:
        cfg = self._config

        # TODO: create a dedicated dataset class with domain split into dedicated folders.
        self._dataset = data.FFHQDataset(cfg.dataset_path)
        self._data_loader = DataLoader(
            dataset=self._dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )

    def _create_models(self) -> None:
        cfg = self._config
        device = self._device

        self._generator = models.Generator(
            style_code_dim=cfg.style_code_dim,
        )

        self._generator_ema = models.Generator(
            style_code_dim=cfg.style_code_dim,
        )

        self._generator_ema.load_state_dict(self._generator.state_dict())

        self._mapping = models.Mapping(
            latent_dim=cfg.mapper_latent_code_dim,
            hidden_dim=cfg.mapper_hidden_dim,
            out_dim=cfg.style_code_dim,
            num_shared_layers=cfg.mapper_shared_layers,
            num_heads=cfg.num_domains,
        )

        self._mapping_ema = models.Mapping(
            latent_dim=cfg.mapper_latent_code_dim,
            hidden_dim=cfg.mapper_hidden_dim,
            out_dim=cfg.style_code_dim,
            num_shared_layers=cfg.mapper_shared_layers,
            num_heads=cfg.num_domains,
        )

        self._mapping_ema.load_state_dict(self._mapping.state_dict())

        self._style_encoder = models.StyleEncoder(
            style_code_dim=cfg.style_code_dim,
            num_heads=cfg.num_domains,
        )

        self._style_encoder_ema = models.StyleEncoder(
            style_code_dim=cfg.style_code_dim,
            num_heads=cfg.num_domains,
        )

        self._style_encoder_ema.load_state_dict(self._style_encoder.state_dict())

        self._discriminator = models.Discriminator(
            num_heads=cfg.num_domains,
        )

        self._generator.to(device)
        self._generator_ema.eval().to(device)
        self._mapping.to(device)
        self._mapping_ema.eval().to(device)
        self._style_encoder.to(device)
        self._style_encoder_ema.eval().to(device)
        self._discriminator.to(device)

    def _create_losses(self) -> None:
        self._adv_loss = losses.AdversarialLoss(self._discriminator)
        self._style_recon_loss = losses.StyleReconstructionLoss(self._style_encoder)
        self._style_diverse_loss = losses.StyleDiversificationLoss()
        self._cycle_loss = losses.CycleConsistencyLoss(self._style_encoder, self._generator)

        device = self._device
        self._adv_loss.to(device)
        self._style_recon_loss.to(device)
        self._style_diverse_loss.to(device)
        self._cycle_loss.to(device)

    def _create_optimizers(self) -> None:
        cfg = self._config

        self._opt_generator = torch.optim.Adam(
            params=self._generator.parameters(),
            lr=cfg.lr_generator,
            betas=(cfg.adam_beta1, cfg.adam_beta2)
        )

        self._opt_mapping = torch.optim.Adam(
            params=self._mapping.parameters(),
            lr=cfg.lr_mapping,
            betas=(cfg.adam_beta1, cfg.adam_beta2)
        )

        self._opt_style_encoder = torch.optim.Adam(
            params=self._style_encoder.parameters(),
            lr=cfg.lr_style_encoder,
            betas=(cfg.adam_beta1, cfg.adam_beta2)
        )

        self._opt_discriminator = torch.optim.Adam(
            params=self._discriminator.parameters(),
            lr=cfg.lr_discriminator,
            betas=(cfg.adam_beta1, cfg.adam_beta2)
        )

    def _create_outputs_dir(self) -> None:
        self._model_dir = 'runs'
        self._samples_dir = os.path.join(self._model_dir, 'samples')
        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._samples_dir, exist_ok=True)

    def _create_tensorboard_writer(self) -> None:
        self._summary_writer = SummaryWriter(log_dir=self._model_dir)

    def _load_models(self, path: str):
        state = torch.load(path)
        self._generator.load_state_dict(state['generator'])
        self._generator_ema.load_state_dict(state['generator_ema'])
        self._mapping.load_state_dict(state['mapping'])
        self._mapping_ema.load_state_dict(state['mapping_ema'])
        self._style_encoder.load_state_dict(state['style_encoder'])
        self._style_encoder_ema.load_state_dict(state['style_encoder_ema'])
        self._discriminator.load_state_dict(state['discriminator'])
        self._opt_generator.load_state_dict(state['opt_generator'])
        self._opt_mapping.load_state_dict(state['opt_mapping'])
        self._opt_style_encoder.load_state_dict(state['opt_style_encoder'])
        self._opt_discriminator.load_state_dict(state['opt_discriminator'])
        self._curr_iter = state['curr_iter']

    def _update_ema_model(self, model: torch.nn.Module, ema_model: torch.nn.Module) -> None:
        ema_beta = self._config.ema_beta
        model_params = dict(model.named_parameters())
        ema_model_params = dict(ema_model.named_parameters())
        for key in model_params:
            ema_model_params[key].data.mul_(ema_beta).add_(1 - ema_beta, model_params[key].data)

    def _update_ema_models(self) -> None:
        self._update_ema_model(self._generator, self._generator_ema)
        self._update_ema_model(self._style_encoder, self._style_encoder_ema)
        self._update_ema_model(self._mapping, self._mapping_ema)

    def _save_models(self, path: str) -> None:
        state = {
            'generator': self._generator.state_dict(),
            'generator_ema': self._generator_ema.state_dict(),
            'mapping': self._mapping.state_dict(),
            'mapping_ema': self._mapping_ema.state_dict(),
            'style_encoder': self._style_encoder.state_dict(),
            'style_encoder_ema': self._style_encoder_ema.state_dict(),
            'discriminator': self._discriminator.state_dict(),
            'opt_generator': self._opt_generator.state_dict(),
            'opt_mapping': self._opt_mapping.state_dict(),
            'opt_style_encoder': self._opt_style_encoder.state_dict(),
            'opt_discriminator': self._opt_discriminator.state_dict(),
            'curr_iter': self._curr_iter,
        }
        torch.save(state, path)


def main():
    if len(sys.argv) != 2:
        print('Usage: train.py [CONFIG_NAME]')

    config_name = sys.argv[1]
    Trainer(config_name).train()


if __name__ == '__main__':
    main()
