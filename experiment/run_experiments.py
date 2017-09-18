import os
import sys
import yaml

with open('../python/local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
main_dir = local_config['results_dir']
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

sys.path.insert(0, os.path.abspath('../python'))
from experiment.cw_inspired_experiment import CWInspiredExperiment
from experiment.base_experiment import BaseExperiment
from experiment.single_universal_experiment import SingleUniversalExperiment
from experiment.selective_unviersal_experiment import SelectiveUniversalExperiment

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


CWInspiredExperiment(
    ensemble=['adv_inception_resnet_v2','inception_v3_tf', 'adv_inception_v3'],
    ensemble_weights=[4.0, 1.0, 1.0],
    targeted=True,
    no_augmentation=True,
    lr=0.32,
    n_iter=27
).run()

SingleUniversalExperiment('crt.npy').run()

SelectiveUniversalExperiment(npy_files=['crt.npy', 'jigsaw.npy'],ensemble=['inception_v3_tf'],ensemble_weights=[1.0]).run()

complete_remaining()