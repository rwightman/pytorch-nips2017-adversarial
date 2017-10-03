import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


def multi_loss(output, target, target_adv=None, is_adv=None, criterion=nn.NLLLoss().cuda()):
    #FIXME use dicts once DataParallel supports it
    loss = criterion(output[0], target)
    #if len(output) > 1 and target_adv is not None:
    #    loss += 0.1 * criterion(output[1], target_adv)
    if len(output) > 1 and is_adv is not None:
        loss += 0.1 * criterion(output[1], is_adv)
    return loss


class MultiTask(nn.Module):

    def __init__(self, model, use_adv_classif=True, use_is_adv=True):
        super(MultiTask, self).__init__()
        self.model = model
        self.in_features = model.num_features

        if use_adv_classif:
            self.classif_adv_class = deepcopy(model.get_classifier())
        else:
            self.classif_adv_class = None

        if use_is_adv:
            self.classif_is_adv = nn.Linear(self.in_features, 2)
        else:
            self.classif_is_adv = None

    def forward(self, x):
        features = self.model.forward_features(x, pool=True)
        output_true = self.model.forward_classifier(features)
        output_multi = OrderedDict({'class_true': output_true})

        if self.classif_adv_class is not None:
            output_adv = self.classif_adv_class(features)
            output_multi['class_adv'] = output_adv

        if self.classif_is_adv is not None:
            output_is_adv = self.classif_is_adv(features)
            output_multi['is_adv'] = output_is_adv

        #FIXME hack to make this work with current lack of dict support in DataParallel
        return tuple(output_multi.values())

    def classifier_params(self):
        params = []
        params += self.model.get_classifier().parameters()
        if self.classif_adv_class is not None:
            params += self.classif_adv_class.parameters()
        if self.classif_is_adv is not None:
            params += self.classif_is_adv.parameters()
        return params


class MultiTaskEnsemble(nn.Module):

    def __init__(
            self,
            models,
            use_features=False,
            use_adv_classif=False,
            use_is_adv=True,
            single_output=False,
            activation_fn=torch.nn.ELU()):
        super(MultiTaskEnsemble, self).__init__()
        self.use_features = use_features
        self.activation_fn = activation_fn
        self.single_output = single_output
        self.num_classes = 1000
        self.models = models if isinstance(models, nn.ModuleList) else nn.ModuleList(models)

        if use_features:
            self.in_features = 0
            for m in models:
                self.in_features += m.num_features
        else:
            self.in_features = len(self.models) * self.num_classes
        self.classif_true_class = nn.Linear(self.in_features, self.num_classes)

        if not use_features:
            fact = 1. / len(self.models)
            wc = torch.cat([torch.eye(self.num_classes, self.num_classes) * fact] * len(self.models), dim=1)
            self.classif_true_class.weight.data = wc

        if use_adv_classif:
            if use_features:
                self.classif_adv_class = nn.Linear(self.in_features, self.num_classes)
            else:
                self.classif_adv_class = deepcopy(self.classif_true_class)
        else:
            self.classif_adv_class = None

        if use_is_adv:
            self.classif_is_adv = nn.Linear(self.in_features, 2)
        else:
            self.classif_is_adv = None

    def forward(self, x):
        if self.use_features:
            outputs = [model.forward_features(x) for model in self.models]
        else:
            outputs = [model(x) for model in self.models]
        output = torch.cat(outputs, 1)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        output_true = self.classif_true_class(output)
        output_multi = OrderedDict({'class_true': output_true})

        if self.single_output:
            return output_true

        if self.classif_adv_class is not None:
            output_adv = self.classif_adv_class(output)
            output_multi['class_adv'] = output_adv

        if self.classif_is_adv is not None:
            output_is_adv = self.classif_is_adv(output)
            output_multi['is_adv'] = output_is_adv

        #FIXME hack to make this work with current lack of dict support in data parallel

        return tuple(output_multi.values())

    def classifier_params(self):
        params = []
        params += self.classif_true_class.parameters()
        if self.classif_adv_class is not None:
            params += self.classif_adv_class.parameters()
        if self.classif_is_adv is not None:
            params += self.classif_is_adv.parameters()
        return params