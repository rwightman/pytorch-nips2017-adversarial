import math
import torch
import numpy as np

def preprocess_gradients(x):
    p = 10
    eps = 1e-6

    log = torch.log(torch.abs(x) + eps)
    clamped_log = torch.clamp(log/p, min=-1.0)
    sign = torch.clamp(x * np.exp(p), min=-1.0, max=1.0)
    return clamped_log, sign
