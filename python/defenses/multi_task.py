import torch
import torch.nn as nn
from copy import deepcopy


def multi_loss(output, target, criterion=nn.NLLLoss().cuda()):
    output_orig, output_adv, output_is_adv = output
    target_orig, target_adv, target_is_adv = target

    loss_orig = criterion(output_orig, target_orig)
    loss_adv = criterion(output_adv, target_adv)
    loss_is_adv = criterion(output_is_adv, target_is_adv)

    return loss_orig + .1 * loss_adv + .1 * loss_is_adv


class MultiTask(nn.Module):

    def __init__(self, model):
        self.model = model
        self.classif_adv_class = deepcopy(model.get_classifier())
        self.classif_adv_type = nn.Linear(self.classif_adv_class.in_features, 1)

    def forward(self, x):
        feat_x = self.model.forward_features(x, pool=True)
        classif_orig = self.model.forward_classifier(feat_x)
        classif_adv = self.classif_adv_class(feat_x)
        is_adv = self.classif_is_adv(feat_x)
        return classif_orig, classif_adv, is_adv



class MultiTaskEnsemble(nn.Module):

    def __init__(self, models, use_features=False):
        self.use_features = use_features
        self.num_classes = 1000
        self.models = nn.ModuleList(models)
        if use_features:
            self.in_features = 0
            for m in models:
                self.in_features += m.num_features
        else:
            self.in_features = len(self.models)*self.num_classes
        self.classif_org_class = nn.Linear(self.in_features, self.num_classes)
        self.classif_adv_class = nn.Linear(self.in_features, self.num_classes)
        self.classif_adv_type = nn.Linear(self.in_features, 1)

    def forward(self, x):
        if self.use_features:
            outputs = [model.forward_features(x) for model in self.models]
        else:
            outputs = [model(x) for model in self.models]
        output = torch.cat(outputs, 1)
        output_org = self.classif_org_class(output)
        output_adv = self.classif_adv_class(output)
        output_ias_adv = self.classif_adv_type(output)
        return output_org, output_adv, output_ias_adv