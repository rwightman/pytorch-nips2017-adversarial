import torch
import torch.nn as nn
import torch.autograd as autograd


def get_normalizer(name):
    if name == 'le':
        return NormalizeLe()
    elif name == 'dpn':
        return NormalizeDpn()
    elif name == 'torchvision':
        return NormalizeTorchvision()
    else:
        assert False, 'Error: Unknown normalizer specified'


class NormalizeLe(nn.Module):
    """Normalize to -1..1 in Google Inception style
    """
    def __init__(self):
        super(NormalizeLe, self).__init__()

    def forward(self, x):
        return (x - 0.5) * 2.0


class NormalizeTorchvision(nn.Module):

    def __init__(self):
        super(NormalizeTorchvision, self).__init__()
        self.mean = autograd.Variable(
            torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda())
        self.std = autograd.Variable(
            torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda())

    def forward(self, x):
        return (x - self.mean) / self.std


class NormalizeDpn(nn.Module):

    def __init__(self):
        super(NormalizeDpn, self).__init__()
        self.mean = autograd.Variable(
            torch.FloatTensor([124.0/255, 117.0/255, 104.0/255]).view(-1, 1, 1).cuda())
        self.scale = 4.2585

    def forward(self, x):
        return (x - self.mean) * self.scale

