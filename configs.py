
class TrainConfig:
    dataset_path: str
    mapper_latent_code_dim: int
    mapper_hidden_dim: int
    mapper_shared_layers: int
    num_domains: int
    style_code_dim: int
    batch_size: int
    training_iterations: int
    lambda_r1: float
    lambda_sty: float
    lambda_ds: float
    lambda_cyc: float
    adam_beta1: float
    adam_beta2: float
    lr_generator: float
    lr_discriminator: float
    lr_style_encoder: float
    lr_mapping: float

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
    dataset_path = '/home/ubuntu/data/ffhq-256'
    mapper_latent_code_dim = 16
    mapper_hidden_dim = 512
    mapper_shared_layers = 7
    num_domains = 2
    style_code_dim = 64
    batch_size = 4  # TODO: increase?
    training_iterations = 500_000
    lambda_r1 = 1.0
    lambda_sty = 1.0
    lambda_ds = 1.0
    lambda_cyc = 1.0
    adam_beta1 = 0.0
    adam_beta2 = 0.99
    lr_generator = 1e-4
    lr_discriminator = 1e-4
    lr_style_encoder = 1e-4
    lr_mapping = 1e-6
