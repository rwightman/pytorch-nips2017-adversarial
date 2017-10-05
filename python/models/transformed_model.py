import torch.nn as nn
from collections import OrderedDict
import processing


def get_output_fn(output_fn='softmax'):
    output_fn = output_fn.lower()
    if output_fn == 'softmax':
        return nn.Softmax()
    elif output_fn == 'log_softmax' or output_fn == 'logsoftmax':
        return nn.LogSoftmax()
    elif output_fn == 'sigmoid':
        return nn.Sigmoid()
    elif output_fn == 'log_sigmoid' or output_fn == 'logsigmoid':
        return nn.LogSigmoid()
    else:
        assert False, 'Error: unknown output_fn specified'


class TransformedModel(nn.Module):

    def __init__(
            self,
            model,
            normalizer='torchvision',
            input_size=299,
            output_fn='softmax',
            drop_first_class=False):
        super(TransformedModel, self).__init__()
        self.model = model
        self.drop_first_class = drop_first_class

        pre = OrderedDict()
        if input_size:
            pre['resize'] = processing.Resize(input_size)
        if normalizer:
            if isinstance(normalizer, str):
                pre['normalize'] = processing.get_normalizer(normalizer)
            elif isinstance(normalizer, nn.Module):
                pre['normalize'] = normalizer
            else:
                raise ValueError('Must be a string or a nn.Module')
        self.pre = nn.Sequential(pre) if pre else None

        post = OrderedDict()
        if output_fn:
            post['output_fn'] = get_output_fn(output_fn)
        self.post = nn.Sequential(post) if post else None

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        o = self.model(x)
        if self.drop_first_class:
            o = o[:, 1:]
        if self.post is not None:
            o = self.post(o)
        return o

    def get_core_model(self):
        return self.model

