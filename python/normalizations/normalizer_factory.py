from . import *


def get_normalizer(normalizer_name):
    if normalizer_name == 'dualpathnet':
        normalizer = DualPathNet()
    elif normalizer_name == 'torchvision':
        normalizer = Torchvision()
    elif normalizer_name == 'le':
        normalizer = Le()
    return normalizer
