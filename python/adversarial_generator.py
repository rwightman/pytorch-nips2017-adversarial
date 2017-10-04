import torch
import torch.utils.data
import numpy as np
from copy import deepcopy

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
        augmentation = augmentation.cuda()
        attack = CWInspired(model, augmentation, **cfg)
    elif attack_name == 'selective_universal':
        attack = SelectiveUniversal(model, **cfg)
    else:
        assert False, 'Unknown attack'
    return attack


class AdversarialGenerator:

    def __init__(self, loader, output_batch_size=8, input_devices=[0], master_output_device=None):
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
            {'models': ['inception_resnet_v2', 'resnet34'], 'weights': [1.0, 1.0]},
            {'models': ['adv_inception_resnet_v2', 'inception_v3_tf']},
        ]
        self.max_epsilons = np.array([8., 12., 16.])
        self.max_epsilon_probs = None
        self.models = []
        self.model_idx = None
        self.target_model = None
        self.attack_idx = 0
        self.input_batch_size = loader.batch_size
        self.output_batch_size = output_batch_size
        self.input_devices = input_devices
        self.master_input_device = input_devices[0]
        self.master_output_device = master_output_device
        self.normal_sample_ratio = 0.5

        self._load_models()

    def _load_models(self):
        # pre-load all model params into system (CPU) memory
        for mc in self.model_cfgs:
            cfgs = [config_from_string(x) for x in mc['models']]
            weights = mc['weights'] if 'weights' in mc and len(mc['weights']) else None
            ensemble = create_ensemble(cfgs, weights)
            self.models.append(ensemble)

    def _next_model(self):
        next_idx = inc_roll(self.model_idx, len(self.models))

        if self.model_idx != next_idx:
            # delete current target_model with params on GPU
            if self.target_model is not None:
                del self.target_model

            # deep copy next model params from CPU to CPU
            model = deepcopy(self.models[next_idx])

            # move next model params to GPU
            if len(self.input_devices) > 1:
                model = torch.nn.DataParallel(model, self.input_devices).cuda()
            else:
                model.cuda(self.master_input_device)

            self.model_idx = next_idx
            self.target_model = model

    def _next_attack(self):
        self._next_model()  # FIXME we could change the model every n attacks?
        attack_idx = np.random.choice(range(len(self.attack_cfgs)), p=self.attack_probs)
        cfg = deepcopy(self.attack_cfgs[attack_idx])
        if 'max_epsilon' not in cfg:
            cfg['max_epsilon'] = np.random.choice(self.max_epsilons, p=self.max_epsilon_probs)
        with torch.cuda.device(self.master_input_device):
            attack = attack_factory(self.target_model, cfg)
        return attack

    def _initialize_outputs(self, image_shape=(3, 299, 299)):
        output_image = torch.zeros((self.output_batch_size,) + image_shape)
        output_target_true = torch.zeros((self.output_batch_size,)).long()
        output_target_adv = torch.zeros((self.output_batch_size,)).long()
        output_is_adv = torch.zeros((self.output_batch_size,)).long()
        if self.master_output_device is not None:
            output_image = output_image.cuda(self.master_output_device)
            output_target_true = output_target_true.cuda(self.master_output_device)
            output_target_adv = output_target_adv.cuda(self.master_output_device)
            output_is_adv = output_is_adv.cuda(self.master_output_device)
        return output_image, output_target_true, output_target_adv, output_is_adv

    def _output_factor(self, curr_batch_size=None):
        return max(curr_batch_size or self.input_batch_size, self.output_batch_size) // self.output_batch_size

    def _input_factor(self):
        return max(self.input_batch_size, self.output_batch_size) // self.input_batch_size

    def __iter__(self):
        out_idx = 0
        images, target_true, target_adv, is_adv = None, None, None, None
        attack = self._next_attack()
        for i, (input, target) in enumerate(self.loader):
            curr_input_batch_size = input.size(0)
            if images is None:
                # lazy creation of output tensors based on input dimensions
                image_shape = input.size()[1:]
                images, target_true, target_adv, is_adv = self._initialize_outputs(image_shape)
            in_idx = 0
            for j in range(self._output_factor(curr_input_batch_size)):
                # copy unperturbed samples from input to output
                num_u = round(curr_input_batch_size * self.normal_sample_ratio)
                images[out_idx:out_idx + num_u, :, :, :] = input[in_idx:in_idx + num_u, :, :, :]
                target_true[out_idx:out_idx + num_u] = target[in_idx:in_idx + num_u]
                target_adv[out_idx:out_idx + num_u] = target[in_idx:in_idx + num_u]
                is_adv[out_idx:out_idx + num_u] = 0
                out_idx += num_u
                in_idx += num_u

                # compute perturbed samples for current attack and copy to output
                num_p = curr_input_batch_size - num_u
                with torch.cuda.device(self.master_input_device):
                    perturbed, adv_targets, _ = attack(
                        input[in_idx:in_idx + num_p, :, :, :].cuda(),
                        target[in_idx:in_idx + num_p].cuda(),
                        batch_idx=i,
                        deadline_time=None)
                if adv_targets is None:
                    adv_targets = target[in_idx:in_idx + num_p]

                images[out_idx:out_idx + num_p, :, :, :] = perturbed
                target_true[out_idx:out_idx + num_p] = target[in_idx:in_idx + num_p]
                target_adv[out_idx:out_idx + num_p] = adv_targets
                is_adv[out_idx:out_idx + num_p] = 1
                out_idx += num_p
                in_idx += num_p

                assert out_idx <= self.output_batch_size
                assert in_idx <= input.size(0)

                if out_idx >= self.output_batch_size:
                    break

            if out_idx == self.output_batch_size:
                yield images, target_true, target_adv, is_adv

                # Initialize next output batch and next attack
                out_idx = 0
                images, target_true, target_adv, is_adv = None, None, None, None
                del attack
                attack = self._next_attack()

        if out_idx and out_idx != self.output_batch_size:
            yield images[:out_idx, :, :, :], target_true[:out_idx], target_adv[:out_idx], is_adv[:out_idx]

    def __len__(self):
        return len(self.loader) * self._input_factor()


def inc_roll(index, length=1):
    if index is None:
        return 0
    else:
        return (index + 1) % length
