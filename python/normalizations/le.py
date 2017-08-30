from .normalizer import Normalizer


class Le(Normalizer):
    def __init__(self):
        super(Le, self).__init__()

    def forward(self, x):
        normalized = x * 2.0 - 1.0
        return super(Le, self).forward(normalized)