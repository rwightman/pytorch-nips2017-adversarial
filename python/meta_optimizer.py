from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from utils import preprocess_gradients
from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D

class MetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, lstm_input_size, lstm_hidden_size):
        super(MetaOptimizer, self).__init__()
        self.meta_model = model
        self.num_layers = num_layers
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size

        self.linear1 = nn.Linear(4, lstm_input_size)
        self.bn = nn.BatchNorm1d(lstm_input_size)

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=num_layers)
        self.hidden = self.init_hidden()

        self.linear2 = nn.Linear(lstm_hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.lstm_hidden_size)),
                Variable(torch.zeros(1, 1, self.lstm_hidden_size)))

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.hidden = (Variable(self.hidden[0].data),
                           Variable(self.hidden[1].data))
        else:
            self.hidden = self.init_hidden()

    def forward(self, x):
        # Gradients preprocessing

        # x is (batch = n_params, features = 4)

        x = self.linear1(x)
        # (batch = n_params, features = 4)

        three_xs = torch.split(x, 299*299, 0)
        three_xs = [ self.bn(xi) for xi in three_xs ]
        x = torch.cat(three_xs)
        # (batch = n_params, features = 4)

        x = F.tanh(x)
        # (batch = n_params, features = 4)

        # Reshape for LSTM
        x = torch.unsqueeze(x, 0)
        # (sequence = 1, batch = n_params, features = 4)

        # Run the LSTM
        x, self.hidden = self.lstm(x)
        # (sequence = 1, batch = n_params, features = 4)

        # Reshape for linear layer
        x = torch.squeeze(x)
        # (batch = n_params, features = 4)

        # Linear out
        x = self.linear2(x)
        # (batch = n_params, features = 1)

        return x.squeeze()

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        flat_params = self.meta_model.get_flat_params()
        flat_grads1, flat_grads2 = preprocess_gradients(model_with_grads.parameters()[0].grad.data.view(-1))

        inputs = Variable(torch.transpose(
            torch.stack((flat_grads1, flat_grads2, flat_params.data, loss.expand_as(flat_grads1))),
            0, 1
        ))

        # Meta update itself
        update = self(inputs)
        flat_params = flat_params + update

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

class FastMetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(FastMetaOptimizer, self).__init__()
        self.meta_model = model

        self.linear1 = nn.Linear(6, 2)
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        # Gradients preprocessing
        x = F.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))
            if use_cuda:
                self.f = self.f.cuda()
                self.i = self.i.cuda()

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        flat_params = self.meta_model.get_flat_params()
        flat_grads = model_with_grads.parameters()[0].grad.data.view(-1)

        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
        print(flat_grads.size())
        print(flat_params.data.size())
        print(loss.size())

        inputs = Variable(torch.cat((preprocess_gradients(flat_grads), flat_params.data, loss)))
        inputs = torch.cat((inputs, self.f, self.i))
        self.f, self.i = self(inputs)

        # Meta update itself
        flat_params = self.f * flat_params - self.i * Variable(flat_grads)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.


class MetaModel:

    def __init__(self, model):
        self.model = model

    def reset(self):
        for module in self.model.children():
            for k in module._parameters.keys():
                module._parameters[k] = Variable(module._parameters[k].data)

    def get_flat_params(self):
        return self.model.parameters()[0].view(-1)

    def set_flat_params(self, flat_params):
        self.model.parameters()[0].data = flat_params.data.view(self.model.parameters()[0].size())

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
