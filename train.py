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
        num_workers=8,  # TODO: increase.
    )

    generator = models.Generator(
        style_code_dim=config.style_code_dim,
    )

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

    if os.path.exists('generator.pt'):
        print('Loading Generator from PT file')
        g_state = torch.load('generator.pt')
        generator.load_state_dict(g_state)

    if os.path.exists('mapping.pt'):
        print('Loading Mapping network from PT file')
        mapping_state = torch.load('mapping.pt')
        mapping.load_state_dict(mapping_state)

    if os.path.exists('style_encoder.pt'):
        print('Loading Style Encoder from PT file')
        encoder_state = torch.load('style_encoder.pt')
        style_encoder.load_state_dict(encoder_state)

    if os.path.exists('discriminator.pt'):
        print('Loading Discriminator from PT file')
        discriminator_state = torch.load('discriminator.pt')
        discriminator.load_state_dict(discriminator_state)

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

    curr_iter = 0
    while True:
        if curr_iter >= config.training_iterations:
            break

        for batch in data_loader:
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

            if (curr_iter + 1) % 100 == 0:
                print('Saving example image...')
                torchvision.utils.save_image(
                    tensor=x_fake,
                    filename='example.jpg',
                    normalize=True,
                    range=(-1, 1)
                )

            if (curr_iter + 1) % 1000 == 0:
                print('Saving models...')
                torch.save(generator.state_dict(), 'generator.pt')
                torch.save(mapping.state_dict(), 'mapping.pt')
                torch.save(style_encoder.state_dict(), 'style_encoder.pt')
                torch.save(discriminator.state_dict(), 'discriminator.pt')

            curr_iter += 1


if __name__ == '__main__':
    fire.Fire(train)
