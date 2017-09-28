import torch
import torch.utils.data
import numpy as np
from copy import deepcopy
from torchvision import datasets
from torchvision import utils

from models import create_ensemble, create_model
from models.model_configs import config_from_string
from attacks.iterative import AttackIterative
from attacks.cw_inspired import CWInspired
from attacks.selective_universal import SelectiveUniversal
import processing


def attack_factory(model, cfg):
    cfg = deepcopy(cfg)
    attack_name = cfg.pop('attack_name')
    print('Creating attack (%s), with args: ' % attack_name, cfg)
    if attack_name == 'iterative':
        attack = AttackIterative(model, **cfg)
    elif attack_name == 'cw_inspired':
        augmentation = processing.build_anp_augmentation_module()
        augmentation = augmentation.cuda(0)
        attack = CWInspired(model, augmentation, **cfg)
    elif attack_name == 'selective_universal':
        attack = SelectiveUniversal(model, **cfg)
    else:
        assert False, 'Unknown attack'
    return attack


class AdversarialGenerator:

    def __init__(self, loader, output_batch_size=8, input_device=0, output_device=None):
        self.loader = loader
        self.attack_cfgs = [
            {'attack_name': 'iterative', 'targeted': True, 'num_steps': 10, 'target_rand': True},
            {'attack_name': 'iterative', 'targeted': False, 'num_steps': 1, 'random_start': True},
            {'attack_name': 'cw_inspired', 'targeted': True, 'n_iter': 38},
            {'attack_name': 'cw_inspired', 'targeted': False, 'n_iter': 38},
        ]
        self.attack_probs = [0.4, 0.4, 0.1, 0.1]
        self.model_cfgs = [  # FIXME these are currently just test configs, need to setup properly
            {'models': ['inception_v3_tf']},
            {'models': ['inception_resnet_v2', 'resnet34'], 'weights': [1.0, .9]},
            {'models': ['adv_inception_resnet_v2', 'inception_v3_tf']},
        ]
        self.max_epsilons = np.array([8., 12., 16.])
        self.max_epsilon_probs = None
        self.models = []
        self.model_idx = None
        self.attack_idx = 0
        self.input_batch_size = loader.batch_size
        self.output_batch_size = output_batch_size
        self.input_device = input_device
        self.output_device = output_device[0] if isinstance(output_device, list) else output_device
        self.normal_sample_ratio = 0.5
        self.img_size = 299

        self._load_models()

    def _load_models(self):
        for mc in self.model_cfgs:
            # pre-load all model params into system (CPU) memory
            cfgs = [config_from_string(x) for x in mc['models']]
            weights = mc['weights'] if 'weights' in mc and len(mc['weights']) else None
            ensemble = create_ensemble(cfgs, weights)
            self.models.append(ensemble)

    def _next_model(self):
        if self.model_idx is not None:
            self.models[self.model_idx].cpu()  # put model params back on CPU
        self.model_idx = inc_roll(self.model_idx, len(self.models))
        model = self.models[self.model_idx]
        # move model params to GPU(s)
        if isinstance(self.input_device, list):
            model = torch.nn.DataParallel(model, self.input_device).cuda()
        else:
            model.cuda(self.input_device)
        return model

    def _next_attack(self, model):
        attack_idx = np.random.choice(range(len(self.attack_cfgs)), p=self.attack_probs)
        cfg = self.attack_cfgs[attack_idx]
        if not 'max_epsilon' in cfg:
            cfg['max_epsilon'] = np.random.choice(self.max_epsilons, p=self.max_epsilon_probs)
        attack = attack_factory(model, cfg)
        return attack

    def _initialize_outputs(self):
        output_image = torch.zeros((self.output_batch_size, 3, self.img_size, self.img_size))
        output_true_target = torch.zeros((self.output_batch_size,)).long()
        output_attack_target = torch.zeros((self.output_batch_size,)).long()
        if self.output_device is not None:
            output_image = output_image.cuda(self.output_device)
            output_true_target = output_true_target.cuda(self.output_device)
            output_attack_target = output_attack_target.cuda(self.output_device)
        return output_image, output_true_target, output_attack_target

    def _output_factor(self, curr_batch_size=None):
        return max(curr_batch_size or self.input_batch_size, self.output_batch_size) // self.output_batch_size

    def _input_factor(self):
        return max(self.input_batch_size, self.output_batch_size) // self.input_batch_size

    def __iter__(self):
        images, target_true, target_attack = self._initialize_outputs()
        out_idx = 0
        model = self._next_model()
        attack = self._next_attack(model)
        for i, (input, target) in enumerate(self.loader):
            curr_input_batch_size = input.size(0)
            in_idx = 0
            for j in range(self._output_factor(curr_input_batch_size)):
                # copy unperturbed samples from input to output
                num_u = round(curr_input_batch_size * self.normal_sample_ratio)
                images[out_idx:out_idx + num_u, :, :, :] = input[in_idx:in_idx + num_u, :, :, :]
                target_true[out_idx:out_idx + num_u] = target[in_idx:in_idx + num_u]
                target_attack[out_idx:out_idx + num_u] = target[in_idx:in_idx + num_u]
                out_idx += num_u
                in_idx += num_u

                # compute perturbed samples for current attack and copy to output
                num_p = curr_input_batch_size - num_u

                perturbed, adv_targets = attack(
                    input[in_idx:in_idx + num_p, :, :, :].cuda(self.input_device),
                    target[in_idx:in_idx + num_p].cuda(self.input_device),
                    batch_idx=i,
                    deadline_time=None)
                if adv_targets is None:
                    adv_targets = target[in_idx:in_idx + num_p]

                images[out_idx:out_idx + num_p, :, :, :] = perturbed
                target_true[out_idx:out_idx + num_p] = target[in_idx:in_idx + num_p]
                target_attack[out_idx:out_idx + num_p] = adv_targets
                out_idx += num_p
                in_idx += num_p

                assert out_idx <= self.output_batch_size
                assert in_idx <= input.size(0)

                if out_idx >= self.output_batch_size:
                    break

            if out_idx == self.output_batch_size:
                yield images, target_true, target_attack
                images, target_true, target_attack = self._initialize_outputs()
                out_idx = 0
                model = self._next_model()
                del attack
                attack = self._next_attack(model)

        if out_idx != self.output_batch_size:
            print('Output truncated last bach (%d)' % out_idx)
            yield images[:out_idx, :, :, :], target_true[:out_idx], target_attack[:out_idx]

    def __len__(self):
        return len(self.loader) * self._input_factor()


def inc_roll(index, length=1):
    if index is None:
        return 0
    else:
        return (index + 1) % length