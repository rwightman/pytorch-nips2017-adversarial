import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Affine(nn.Module):
    def __init__(self,
                 min_rot, max_rot,
                 min_shear_x, max_shear_x,
                 min_shear_y, max_shear_y,
                 min_scale_x, max_scale_x,
                 min_scale_y, max_scale_y
                 ):
        super(Affine, self).__init__()

        self.min_rot = min_rot
        self.max_rot = max_rot
        self.min_shear_x = min_shear_x
        self.max_shear_x = max_shear_x
        self.min_shear_y = min_shear_y
        self.max_shear_y = max_shear_y
        self.min_scale_x = min_scale_x
        self.max_scale_x = max_scale_x
        self.min_scale_y = min_scale_y
        self.max_scale_y = max_scale_y

    def forward(self, x):

        rot_theta = np.random.uniform(self.min_rot, self.max_rot)
        shear_phi_x = np.random.uniform(self.min_shear_x, self.max_shear_x)
        shear_psi_y = np.random.uniform(self.min_shear_y, self.max_shear_y)
        scale_x = np.random.uniform(self.min_scale_x, self.max_scale_x)
        scale_y = np.random.uniform(self.min_scale_y, self.max_scale_y)

        rotation_matrix = np.array([[np.cos(rot_theta), np.sin(rot_theta), 0],
                                    [-np.sin(rot_theta), np.cos(rot_theta), 0],
                                    [0, 0, 1]], dtype=np.float32)

        shear_matrix = np.array([[1, np.tan(shear_phi_x), 0],
                                 [np.tan(shear_psi_y), 1, 0],
                                 [0, 0, 1]], dtype=np.float32)

        scale_matrix = np.array([[scale_x, 0, 0],
                                 [0, scale_y, 0],
                                 [0, 0, 1]], dtype=np.float32)

        transformation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), scale_matrix)[0:2, :]

        matrix = torch.FloatTensor(np.stack([transformation_matrix for _ in range(x.size(0))])).cuda()

        grid = F.affine_grid(matrix, x.size())

        return F.grid_sample(x, grid)
