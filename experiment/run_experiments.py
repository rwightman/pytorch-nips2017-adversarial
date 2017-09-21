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
    for a in attacks:
        for d in ALL_MODELS:
            BaseExperiment(ensemble=[d], ensemble_weights=[1.0], attack_type='attack', attack_name=a).run()
    targeted_attacks = os.listdir(os.path.join(main_dir, 'targeted_attacks'))
    for a in targeted_attacks:
        for d in ALL_MODELS:
            BaseExperiment(ensemble=[d], ensemble_weights=[1.0], attack_type='targeted_attack', attack_name=a).run()

all_npy_files = [
    "african_chameleon.npy",
    "african_chameleon2.npy",
    "brain_coral.npy",
    "crt.npy",
    "jigsaw.npy",
    "jigsaw2.npy",
    "jigsaw3.npy",
    "jigsaw4.npy",
    "monitor.npy",
    "prayer_rug.npy",
    "spider_web.npy",
    "spider_web2.npy",
    "spider_web3.npy",
    "tv.npy",
]

for f in all_npy_files:
    SingleUniversalExperiment(f).run()
complete_remaining()

CWInspiredExperiment(
    ensemble=['adv_inception_resnet_v2','inception_v3_tf', 'adv_inception_v3'],
    ensemble_weights=[4.0, 1.0, 1.0],
    targeted=True,
    no_augmentation=True,
    lr=0.20,
    n_iter=27
).run()

CWInspiredExperiment(
    ensemble=['adv_inception_resnet_v2','inception_v3_tf', 'resnet34'],
    ensemble_weights=[1.0, 1.0, 1.0],
    targeted=False,
    lr=0.24,
    n_iter=31,
    target_nth_highest=3,
).run()

SelectiveUniversalExperiment(
    npy_files=all_npy_files,
    ensemble=["adv_inception_resnet_v2", "inception_v3_tf"],
    ensemble_weights=["1.0", "1.0"]
).run()
SelectiveUniversalExperiment(
    npy_files=all_npy_files,
    ensemble=["adv_inception_resnet_v2", "inception_v3_tf", "adv_inception_v3"],
    ensemble_weights=["1.0", "1.0", "1.0"]
).run()

complete_remaining()