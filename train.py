import os

import fire
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader

import configs
import data
import losses
import models


def train(config_name: str):
    config: configs.TrainConfig = getattr(configs, config_name, None)
    if config is None:
        raise ValueError(f'configuration "{config_name}" not found in {configs.__file__}')

    print(config.str())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO: create a dedicated dataset class with domain split into dedicated folders.
    dataset = data.FFHQDataset(config.dataset_path)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    generator = models.Generator(
        style_code_dim=config.style_code_dim,
    )

    generator_ema = models.Generator(
        style_code_dim=config.style_code_dim,
    )

    generator_ema.load_state_dict(generator.state_dict())

    mapping = models.Mapping(
        latent_dim=config.mapper_latent_code_dim,
        hidden_dim=config.mapper_hidden_dim,
        out_dim=config.style_code_dim,
        num_shared_layers=config.mapper_shared_layers,
        num_heads=config.num_domains,
    )

    style_encoder = models.StyleEncoder(
        style_code_dim=config.style_code_dim,
        num_heads=config.num_domains,
    )

    discriminator = models.Discriminator(
        num_heads=config.num_domains,
    )

    opt_generator = torch.optim.Adam(
        params=generator.parameters(),
        lr=config.lr_generator,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    opt_mapping = torch.optim.Adam(
        params=mapping.parameters(),
        lr=config.lr_mapping,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    opt_style_encoder = torch.optim.Adam(
        params=style_encoder.parameters(),
        lr=config.lr_style_encoder,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    opt_discriminator = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=config.lr_discriminator,
        betas=(config.adam_beta1, config.adam_beta2)
    )

    if os.path.exists('trained_models/models.pt'):
        print('Loading previously saved models from PT file...')
        state = torch.load('generator.pt')
        generator.load_state_dict(state['generator'])
        generator_ema.load_state_dict(state['generator_ema'])
        mapping.load_state_dict(state['mapping'])
        style_encoder.load_state_dict(state['style_encoder'])
        discriminator.load_state_dict(state['discriminator'])
        opt_generator.load_state_dict(state['opt_generator'])
        opt_mapping.load_state_dict(state['opt_mapping'])
        opt_style_encoder.load_state_dict(state['opt_style_encoder'])
        opt_discriminator.load_state_dict(state['opt_discriminator'])
        curr_iter = state['curr_iter']
    else:
        curr_iter = 0

    adv_loss = losses.AdversarialLoss(discriminator)
    style_recon_loss = losses.StyleReconstructionLoss(style_encoder)
    style_diverse_loss = losses.StyleDiversificationLoss()
    cycle_loss = losses.CycleConsistencyLoss(style_encoder, generator)

    generator.to(device)
    mapping.to(device)
    style_encoder.to(device)
    discriminator.to(device)
    adv_loss.to(device)
    style_recon_loss.to(device)
    style_diverse_loss.to(device)
    cycle_loss.to(device)

    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('samples', exist_ok=True)

    while True:
        for batch in data_loader:
            if curr_iter >= config.training_iterations:
                break

            x_real = batch['image'].to(device)
            y_real = batch['attributes']['gender'].to(device).reshape(-1, 1)

            z = torch.randn((config.batch_size, config.mapper_latent_code_dim)).to(device)
            z_2 = torch.randn((config.batch_size, config.mapper_latent_code_dim)).to(device)
            y_fake = torch.randint(config.num_domains, size=(config.batch_size, 1)).to(device)
            s = mapping(z, y_fake)
            s_2 = mapping(z_2, y_fake)
            x_fake = generator(x_real, s)
            x_fake_2 = generator(x_real, s_2)

            l_adv_d, l_adv_g, r1_reg = adv_loss(x_real, y_real, x_fake, y_fake)
            l_sty = style_recon_loss(x_fake, y_fake, s)
            l_ds = style_diverse_loss(x_fake, x_fake_2)
            l_cyc = cycle_loss(x_real, y_real, x_fake)

            l_d = l_adv_d + config.lambda_r1 * r1_reg
            lambda_ds = max(0.0, config.lambda_ds - curr_iter * config.lambda_ds / config.training_iterations / 2)
            l_fge = (
                l_adv_g
                + config.lambda_sty * l_sty
                - lambda_ds * l_ds
                + config.lambda_cyc * l_cyc
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

            g_params = dict(generator.named_parameters())
            g_ema_params = dict(generator_ema.named_parameters())
            for key in g_params:
                g_ema_params[key].data.mul_(config.ema_beta).add_(1 - config.ema_beta, g_params[key].data)

            l_adv_d_np = l_adv_d.to("cpu").detach().numpy()
            l_adv_g_np = l_adv_g.to("cpu").detach().numpy()
            r1_reg_np = r1_reg.to("cpu").detach().numpy()
            l_sty_np = l_sty.to("cpu").detach().numpy()
            l_ds_np = l_ds.to("cpu").detach().numpy()
            l_cyc_np = l_cyc.to("cpu").detach().numpy()
            l_fge_np = l_fge.to("cpu").detach().numpy()
            l_d_np = l_d.to("cpu").detach().numpy()
            print(f'Iteration: {curr_iter}, '                  
                  f'L_adv_D: {l_adv_d_np:.2f}, '
                  f'L_adv_G: {l_adv_g_np:.2f}, '
                  f'R1_reg: {r1_reg_np:.2f}, '
                  f'L_sty: {l_sty_np:.2f}, '
                  f'L_ds: {l_ds_np:.2f}, '
                  f'L_cyc: {l_cyc_np:.2f}, '
                  f'L_D: {l_d_np:.2f}, '
                  f'L_FGE: {l_fge_np:.2f}')

            if (curr_iter + 1) % 100 == 0 or curr_iter == config.training_iterations - 1:
                print('Saving example image...')
                torchvision.utils.save_image(
                    tensor=torch.cat(x_real, x_fake),
                    filename=f'samples/iter_{curr_iter}.jpg',
                    nrow=config.batch_size,
                    normalize=True,
                    range=(-1, 1)
                )

                x_fake_ema = generator_ema(x_real, s)
                torchvision.utils.save_image(
                    tensor=torch.cat(x_real, x_fake_ema),
                    filename=f'samples/iter_{curr_iter}_ema.jpg',
                    nrow=config.batch_size,
                    normalize=True,
                    range=(-1, 1)
                )

            if (curr_iter + 1) % 1000 == 0 or curr_iter == config.training_iterations - 1:
                print('Saving models...')
                state = {
                    'generator': generator.state_dict(),
                    'generator_ema': generator_ema.state_dict(),
                    'mapping': mapping.state_dict(),
                    'style_encoder': style_encoder.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'opt_generator': opt_generator.state_dict(),
                    'opt_mapping': opt_mapping.state_dict(),
                    'opt_style_encoder': opt_style_encoder.state_dict(),
                    'opt_discriminator': opt_discriminator.state_dict(),
                    'curr_iter': curr_iter,
                }

                torch.save(state, 'trained_models/models.pt')

            curr_iter += 1


if __name__ == '__main__':
    fire.Fire(train)
