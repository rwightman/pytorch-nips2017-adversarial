import torch
import torch.utils.data
import numpy as np
from copy import deepcopy

from models import create_ensemble, create_model
from models.model_configs import config_from_string
from attacks.attack_factory import attack_factory


class AdversarialGenerator:

    def __init__(
            self,
            loader,
            model_cfgs,
            attack_cfgs,
            attack_probs=None,
            output_batch_size=8,
            input_devices=[0],
            master_output_device=None):

        self.loader = loader
        self.model_cfgs = model_cfgs
        self.attack_cfgs = attack_cfgs
        self.attack_probs = attack_probs or [1/len(attack_cfgs)] * len(attack_cfgs)
        self.max_epsilons = np.array([8., 12., 16.])
        self.max_epsilon_probs = None
        self.models = []
        self.model_idx = None
        self.dogfood_model_idx = None
        self.target_model = None  # deep copy of currently targeted model, loaded onto GPU
        self.model_count = 0 # number of model transitions
        self.attack_count = 0 # number of attack transitions

        self.input_batch_size = loader.batch_size
        self.output_batch_size = output_batch_size
        self.input_devices = input_devices
        self.master_input_device = input_devices[0]
        self.master_output_device = master_output_device

        self.normal_sample_ratio = 0.5
        self.model_change_cycle = 5
        self.dogfood_cycle = 2

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
        #next_idx = np.random.randint(0, len(self.models))
        dogfood = False
        if self.dogfood_model_idx is not None:
            c = self.model_count // len(self.models) + 1
            if next_idx == self.dogfood_model_idx:
                if c % self.dogfood_cycle != 0:
                    next_idx = inc_roll(next_idx, len(self.models))
                else:
                    dogfood = True

        # NOTE: dogfood model needs to be recopied on dogfood cycle even
        # if it's the only model as params change
        if dogfood or self.model_idx != next_idx:
            # delete current target_model with params on GPU
            if self.target_model is not None:
                del self.target_model

            # deep copy next model params from CPU to CPU
            # FIXME TBD, if model is dogfood on another GPU, 
            # does deepcopy work or double the defense GPU mem usage before cuda to attack GPU?
            model = deepcopy(self.models[next_idx])

            # move next model params to GPU
            if len(self.input_devices) > 1:
                model = torch.nn.DataParallel(model, self.input_devices).cuda()
            else:
                model.cuda(self.master_input_device)

            self.model_idx = next_idx
            self.target_model = model

        self.model_count += 1

    def _next_attack(self):
        if self.target_model is None or self.attack_count % self.model_change_cycle == 0:
            self._next_model()
        attack_idx = np.random.choice(range(len(self.attack_cfgs)), p=self.attack_probs)
        cfg = deepcopy(self.attack_cfgs[attack_idx])
        if 'max_epsilon' not in cfg:
            cfg['max_epsilon'] = np.random.choice(self.max_epsilons, p=self.max_epsilon_probs)
        with torch.cuda.device(self.master_input_device):
            attack = attack_factory(self.target_model, cfg)
        self.attack_count += 1
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

    def set_dogfood(self, model):
        if self.dogfood_model_idx is not None:
            # only one dogfood model allowed, replace existing
            self.models[self.dogfood_model_idx] = model
        else:
            self.models.append(model)
            self.dogfood_model_idx = len(self.models) - 1


def inc_roll(index, length=1):
    if index is None:
        return 0
    else:
        return (index + 1) % length
