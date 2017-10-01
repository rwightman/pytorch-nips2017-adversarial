import torch
import torch.nn as nn
from copy import deepcopy


def multi_loss(output, target, target_adv=None, is_adv=None, criterion=nn.NLLLoss().cuda()):
    loss = criterion(output['class_true'], target)
    if 'class_adv' in output and target_adv is not None:
        loss += 0.1 * criterion(output['class_adv'], target_adv)
    if 'is_adv' in output and is_adv is not None:
        loss += 0.1 * criterion(output['is_adv'], is_adv)
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
        output = {'class_true': output_true}

        if self.classif_adv_class is not None:
            output_adv = self.classif_adv_class(features)
            output['class_adv'] = output_adv

        if self.classif_is_adv is not None:
            output_is_adv = self.classif_is_adv(features)
            output['is_adv'] = output_is_adv
        return output

    def classifier_params(self):
        params = []
        params.append(self.model.get_classifier().parameters())
        if self.classif_adv_class is not None:
            params.append(self.classif_adv_class.parameters())
        if self.classif_is_adv is not None:
            params.append(self.classif_is_adv.parmeters())
        return params


class MultiTaskEnsemble(nn.Module):

    def __init__(
            self,
            models,
            use_features=False,
            use_adv_classif=True,
            use_is_adv=True,
            activation_fn=torch.nn.ELU()):
        super(MultiTaskEnsemble, self).__init__()
        self.use_features = use_features
        self.activation_fn = activation_fn
        self.num_classes = 1000
        self.models = models if isinstance(models, nn.ModuleList) else nn.ModuleList(models)

        if use_features:
            self.in_features = 0
            for m in models:
                self.in_features += m.num_features
        else:
            self.in_features = len(self.models) * self.num_classes
        self.classif_true_class = nn.Linear(self.in_features, self.num_classes)

        if use_adv_classif:
            self.classif_adv_class = nn.Linear(self.in_features, self.num_classes)
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
        output = {'class_true': output_true}

        if self.classif_adv_class is not None:
            output_adv = self.classif_adv_class(output)
            output['class_adv'] = output_adv

        if self.classif_is_adv is not None:
            output_is_adv = self.classif_is_adv(output)
            output['is_adv'] = output_is_adv
        return output

    def classifier_params(self):
        params = []
        params.append(self.classif_true_class.parameters())
        if self.classif_adv_class is not None:
            params.append(self.classif_adv_class.parameters())
        if self.classif_is_adv is not None:
            params.append(self.classif_is_adv.parmeters())
        return params
