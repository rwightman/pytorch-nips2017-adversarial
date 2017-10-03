import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Shear(nn.Module):
    def __init__(self, LWC_size, shears, probs):
        super(Shear, self).__init__()

        self.shears = shears
        self.probs = probs

        def shear_matrix(strengthx, strengthy):
            return np.array([[1, strengthy, 0],
                             [strengthx, 1, 0]],
                            dtype=np.float32)[None,:,:]

        # Grids are build for a batch of 1 and then used later
        self.grids = [F.affine_grid(torch.FloatTensor(shear_matrix(s[0], s[1])), LWC_size) for s in shears]

    def forward(self, x):
        random = np.random.uniform(0.0,1.0)

        cumulative_prob = 0.0
        for prob, grid in zip(self.probs, self.grids):
            cumulative_prob += prob
            if cumulative_prob > random:
                batch_size = x.size(0)
                useable_grid = grid.expand(torch.Size([batch_size, grid.size(1), grid.size(2), grid.size(3)]))
                return F.grid_sample(x, useable_grid)

