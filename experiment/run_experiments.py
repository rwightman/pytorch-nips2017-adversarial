import os
import sys
import yaml

with open('../python/local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
main_dir = local_config['results_dir']
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

sys.path.insert(0, os.path.abspath('../python'))
from cw_inspired_experiment import CWInspiredExperiment
from base_experiment import BaseExperiment
from single_universal_experiment import SingleUniversalExperiment
from selective_unviersal_experiment import SelectiveUniversalExperiment
from smart_universal import SmartUniversalExperiment
from eps_dependent import EpsilonDependentExperiment

ALL_MODELS = [
    'inception_v3',
    'inception_v3_tf',
    'adv_inception_v3',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'densenet121',
    'densenet169',
    'densenet201',
    'densenet161',
    'adv_inception_resnet_v2',
    'inception_resnet_v2',
    'squeezenet1_0',
    'squeezenet1_1',
    'alexnet',
    'dpn107',
    'dpn92_extra',
    'dpn92',
    'dpn68'
]

# Run all base defenses against existing attacks
def complete_remaining():
    attacks = os.listdir(os.path.join(main_dir, 'attacks'))
    targeted_attacks = os.listdir(os.path.join(main_dir, 'targeted_attacks'))

    for d in ['adv_inception_resnet_v2', 'inception_v3_tf', 'adv_inception_v3']:
        for a in attacks:
            BaseExperiment(ensemble=[d], ensemble_weights=[1.0], attack_type='attack', attack_name=a).run()
        for a in targeted_attacks:
            BaseExperiment(ensemble=[d], ensemble_weights=[1.0], attack_type='targeted_attack', attack_name=a).run()

    for a in attacks:
        for d in ALL_MODELS:
            BaseExperiment(ensemble=[d], ensemble_weights=[1.0], attack_type='attack', attack_name=a).run()

    for a in targeted_attacks:
        for d in ALL_MODELS:
            BaseExperiment(ensemble=[d], ensemble_weights=[1.0], attack_type='targeted_attack', attack_name=a).run()


all_npy_files = [
    "african_chameleon.npy",
    "african_chameleon2.npy",
    "africangray.npy",
    "brain_coral.npy",
    "crt.npy",
    "hand_adv_poc3.npy",
    "honeycomb.npy",
    "honeycomb2.npy",
    "jigsaw.npy",
    "jigsaw2.npy",
    "jigsaw3.npy",
    "jigsaw4.npy",
    "jigsaw5.npy",
    "ladybug.npy",
    "monitor.npy",
    "monitor_manuala.npy",
    "monitor_manualb.npy",
    "prayer_rug.npy",
    "spider_web.npy",
    "spider_web2.npy",
    "spider_web3.npy",
    "tv.npy",
]

def get_round_4_nontargeted(eps):
    return EpsilonDependentExperiment(
        {
            '4': CWInspiredExperiment(
                ensemble=['adv_inception_resnet_v2', 'inception_v3_tf', 'adv_inception_v3'],
                ensemble_weights=[1.0, 1.0, 1.0],
                targeted=False,
                lr=0.20,
                n_iter=27,
                target_nth_highest=3,
                no_augmentation=True
            ),
            '8': SmartUniversalExperiment(
                npy_files=["brain_coral.npy","jigsaw.npy","jigsaw2.npy","jigsaw3.npy","jigsaw4.npy","jigsaw5.npy","monitor.npy","monitor_manuala.npy","monitor_manualb.npy","spider_web.npy","spider_web2.npy","spider_web3.npy",],
                ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
                ensemble_weights=["1.0", "1.0", "1.0"],
                try_mirrors=False,
                no_augmentation=True,
                lr=0.2
            ),
            '12': SmartUniversalExperiment(
                npy_files=["brain_coral.npy","jigsaw.npy","jigsaw2.npy","jigsaw3.npy","jigsaw4.npy","jigsaw5.npy","monitor.npy","monitor_manuala.npy","monitor_manualb.npy","spider_web.npy","spider_web2.npy","spider_web3.npy",],
                ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
                ensemble_weights=["1.0", "1.0", "1.0"],
                try_mirrors=False,
                no_augmentation=True,
                lr=0.2
            ),
            '16': SmartUniversalExperiment(
                npy_files=["brain_coral.npy","jigsaw.npy","jigsaw2.npy","jigsaw3.npy","jigsaw4.npy","jigsaw5.npy","monitor.npy","monitor_manuala.npy","monitor_manualb.npy","spider_web.npy","spider_web2.npy","spider_web3.npy",],
                ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
                ensemble_weights=["1.0", "1.0", "1.0"],
                try_mirrors=False,
                no_augmentation=True,
                lr=0.2
            )
        },
        False,
        'round_4_nontargeted_eps{}'.format(eps),
        max_epsilon=eps,
    )

def get_round_4_targeted(eps):
    return EpsilonDependentExperiment(
        {
            '4': CWInspiredExperiment(
                ensemble=["adv_inception_resnet_v2", "inception_v3_tf"],
                ensemble_weights=["4.0", "1.0"],
                targeted=True,
                lr=0.20,
                n_iter=35,
                no_augmentation=True
            ),
            '8': CWInspiredExperiment(
                ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
                ensemble_weights=["1.0", "2.0", "1.0"],
                targeted=True,
                lr=0.20,
                n_iter=27,
                no_augmentation=True
            ),
            '12': CWInspiredExperiment(
                ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
                ensemble_weights=["4.0", "1.0", "1.0"],
                targeted=True,
                lr=0.20,
                n_iter=27,
                no_augmentation=True
            ),
            '16': CWInspiredExperiment(
                ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
                ensemble_weights=["4.0", "1.0", "1.0"],
                targeted=True,
                lr=0.20,
                n_iter=27,
                no_augmentation=True
            )
        },
        True,
        'round_4_targeted2_eps{}'.format(eps),
        max_epsilon=eps,
    )

get_round_4_nontargeted(0).deploy()
get_round_4_targeted(0).deploy()

for eps in [4, 8, 12, 16]:
    get_round_4_nontargeted(eps).run()
    get_round_4_targeted(eps).run()


for eps in [8, 12, 16]:
    SmartUniversalExperiment(
        npy_files=["honeycomb.npy", "honeycomb2.npy", "jigsaw.npy", "jigsaw2.npy",
                   "jigsaw3.npy", "jigsaw5.npy", "monitor.npy", "monitor_manuala.npy",
                   "monitor_manualb.npy", "spider_web.npy", "spider_web2.npy", "spider_web3.npy"],
        ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
        ensemble_weights=["1.0", "1.0", "1.0"],
        try_mirrors=False,
        no_augmentation=True,
        lr=0.2,
        max_epsilon=eps
    ).run()

CWInspiredExperiment(
    ensemble=['adv_inception_resnet_v2', 'inception_v3_tf', 'adv_inception_v3'],
    ensemble_weights=[1.0, 1.0, 1.0],
    targeted=False,
    lr=0.20,
    n_iter=27,
    target_nth_highest=3,
    no_augmentation=True,
    max_epsilon=4,
    always_target=109).run()

CWInspiredExperiment(
    ensemble=['adv_inception_resnet_v2', 'inception_v3_tf', 'adv_inception_v3'],
    ensemble_weights=[1.0, 1.0, 1.0],
    targeted=False,
    lr=0.20,
    n_iter=27,
    target_nth_highest=3,
    no_augmentation=True,
    max_epsilon=4,
    always_target=611).run()

for eps in [16,12,8,4]:
    SingleUniversalExperiment('hand_adv_poc3.npy',max_epsilon=eps).run()
    BaseExperiment(['adv_inception_resnet_v2'],[1.0],'eps{}_universal_hand_adv_poc3.npy'.format(eps),'attack').run()
    BaseExperiment(['inception_v3_tf'], [1.0], 'eps{}_universal_hand_adv_poc3.npy'.format(eps), 'attack').run()
    BaseExperiment(['adv_inception_v3'], [1.0], 'eps{}_universal_hand_adv_poc3.npy'.format(eps), 'attack').run()

complete_remaining()

for eps in [16,12,8,4]:
    for nf in all_npy_files:
        SingleUniversalExperiment(nf,max_epsilon=eps).run()

complete_remaining()