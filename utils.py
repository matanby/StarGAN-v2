import json
from typing import Optional, Dict

import torch


def is_valid_tensor(x: torch.Tensor,
                    num_dims: Optional[int] = None,
                    batch_size: Optional[int] = None):

    if not isinstance(x, torch.Tensor):
        return False

    if num_dims and len(x.shape) != num_dims:
        return False

    if batch_size and x.shape[0] != batch_size:
        return False

    return True


def is_valid_image_tensor(x: torch.Tensor) -> bool:
    result = is_valid_tensor(x, num_dims=4) and x.shape[1] == 3
    return result


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(obj: Dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f)
