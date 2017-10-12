from copy import deepcopy

from attacks.iterative import AttackIterative
from attacks.cw_inspired import CWInspired
from attacks.selective_universal import SelectiveUniversal
from attacks.restart_attack import RestartAttack

import processing

def attack_factory(model, cfg):
    cfg = deepcopy(cfg)
    attack_name = cfg.pop('attack_name')

    if 'n_restarts' in cfg:
        n_restarts = cfg.pop('n_restarts')
    else:
        n_restarts = None

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

    if n_restarts is not None:
        attack = RestartAttack(attack=attack, n_restarts=n_restarts)

    return attack