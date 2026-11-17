import random

import numpy as np
import torch


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:

    return torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
