
class TrainConfig:
    adam_beta1: float
    adam_beta2: float
    batch_size: int
    dataset_path: str
    ema_beta: float
    lambda_cyc: float
    lambda_ds: float
    lambda_r1: float
    lambda_sty: float
    lr_discriminator: float
    lr_generator: float
    lr_mapping: float
    lr_style_encoder: float
    mapper_hidden_dim: int
    mapper_latent_code_dim: int
    mapper_shared_layers: int
    model_snapshot_interval: int
    num_domains: int
    style_code_dim: int
    tb_losses_log_interval: int
    tb_samples_log_interval: int
    training_iterations: int

    @classmethod
    def str(cls):
        s = ''
        for key, value in cls.__dict__.items():
            if key.startswith('__'):
                continue
            s += f'{key}: {value}\n'

        s = s.strip('\n')
        return s


class FFHQ(TrainConfig):
    adam_beta1 = 0.0
    adam_beta2 = 0.99
    batch_size = 4  # TODO: increase?
    dataset_path = '/home/ubuntu/data/ffhq-256'
    ema_beta = 0.999
    lambda_cyc = 1.0
    lambda_ds = 1.0
    lambda_r1 = 1.0
    lambda_sty = 1.0
    lr_discriminator = 1e-4
    lr_generator = 1e-4
    lr_mapping = 1e-6
    lr_style_encoder = 1e-4
    mapper_hidden_dim = 512
    mapper_latent_code_dim = 16
    mapper_shared_layers = 7
    model_snapshot_interval = 1000
    num_domains = 2
    style_code_dim = 64
    tb_losses_log_interval = 10
    tb_samples_log_interval = 100
    training_iterations = 500_000
