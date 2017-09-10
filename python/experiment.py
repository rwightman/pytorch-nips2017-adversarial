import subprocess
import os
import yaml
from experiments.model_configs import config_from_string

with open('local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
main_dir = local_config['results_dir']
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

CHECKPOINT_DIR = local_config['checkpoints_dir']
IMAGES_DIR = local_config['images_dir']

def validate(experiment_name, targeted):
    validate_cmd = ['python', 'validate_attack_inf_norm.py', experiment_name]
    if targeted:
        validate_cmd.append('--targeted')
    subprocess.call(validate_cmd)

def run_cw_inspired_experiment(
        ensemble,
        ensemble_weights,
        targeted,
        input_dir=os.path.abspath(IMAGES_DIR),
        max_epsilon=16,
        no_augmentation=False,
        no_augmentation_blurring=False,
        n_iter=100,
        lr=0.02,
        target_nth_highest=6,
        gaus_blur_prob=0.5,
        gaus_blur_size=5,
        gaus_blur_sigma=3.0):
    experiment_name = 'cw_inspired_{}'.format(''.join(sorted(ensemble)))
    if no_augmentation:
        experiment_name = '{}_noaug'.format(experiment_name)
    if no_augmentation_blurring:
        experiment_name = '{}_noblur'.format(experiment_name)
    if n_iter != 100:
        experiment_name = '{}_{}iter'.format(experiment_name, n_iter)
    if lr != 0.02:
        experiment_name = '{}_lr{}'.format(experiment_name, lr)
    if target_nth_highest != 6:
        experiment_name = '{}_trg{}'.format(experiment_name, target_nth_highest)

    # Gaus Blur Experiment Variables
    if gaus_blur_prob != 0.5:
        experiment_name = '{}_gbp{}'.format(experiment_name, gaus_blur_prob)
    if gaus_blur_size != 5:
        experiment_name = '{}_gbsiz{}'.format(experiment_name, gaus_blur_size)
    if gaus_blur_sigma != 3.0:
        experiment_name = '{}_gbsig{}'.format(experiment_name, gaus_blur_sigma)

    output_dir = os.path.join(main_dir, 'targeted_attacks' if targeted else 'attacks', experiment_name)
    if not os.path.exists(output_dir):
        print('Running experiment {}: {}attack {}.'.format(experiment_name, 'targeted ' if targeted else '', 'cw_inspired'))

        os.makedirs(output_dir)

        python_cmd = [
            'python',
            'run_cw_inspired.py',
            '--input_dir=/input_images',
            '--output_dir=/output_images',
            '--max_epsilon={}'.format(max_epsilon) ]

        checkpoint_paths = [config_from_string(m)['checkpoint_file'] for m in ensemble]
        python_cmd.append('--checkpoint_paths')
        python_cmd.extend([os.path.join('/checkpoints/',cp) for cp in checkpoint_paths]) # We're going to mount the checkpoint folder below

        if targeted:
            python_cmd.append('--targeted')
        if no_augmentation:
            python_cmd.append('--no_augmentation')
        if no_augmentation_blurring:
            python_cmd.append('--no_augmentation_blurring')
        if n_iter != 100:
            python_cmd.extend(['--n_iter',str(n_iter)])
        if lr != 0.02:
            python_cmd.extend(['--lr', str(lr)])
        if target_nth_highest != 6:
            python_cmd.extend(['--target_nth_highest', str(target_nth_highest)])
        python_cmd.extend(['--gaus_blur_prob', str(gaus_blur_prob), '--gaus_blur_size', str(gaus_blur_size), '--gaus_blur_sigma', str(gaus_blur_sigma)])
        python_cmd.append('--ensemble')
        python_cmd.extend(ensemble)
        python_cmd.append('--ensemble_weights')
        python_cmd.extend([str(e) for e in ensemble_weights])
        python_cmd.extend(['--batch_size','8'])

        cmd = [
            'nvidia-docker','run',
            '-v','{}:/input_images'.format(os.path.abspath(input_dir)),
            '-v','{}:/output_images'.format(os.path.abspath(output_dir)),
            '-v','{}:/code'.format(os.path.abspath(os.getcwd())),
            '-v', '{}:/checkpoints'.format(os.path.abspath(CHECKPOINT_DIR)), # Here is mounting that checkpoint folder
            '-w','/code',
            'rwightman/pytorch-extra'
        ]

        cmd.extend(python_cmd)

        subprocess.call(cmd)

        validate(experiment_name, targeted)

def run_base_defense_experiment(ensemble, ensemble_weights, attack_name, targeted):
    input_dir = os.path.join(main_dir, 'targeted_attacks' if targeted else 'attacks', attack_name)

    experiment_name = 'base_{}'.format(''.join(sorted(ensemble)))
    output_dir = os.path.join(main_dir, 'defenses', experiment_name, 'targeted_attacks' if targeted else 'attacks')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, '{}.csv'.format(attack_name))):
        python_cmd = [
            'python',
            'run_base.py',
            '--input_dir=/input_images',
            '--output_file=/output_data/{}.csv'.format(attack_name)]
        python_cmd.append('--ensemble')
        python_cmd.extend(ensemble)
        python_cmd.append('--ensemble_weights')
        python_cmd.extend([str(e) for e in ensemble_weights])

        checkpoint_paths = [config_from_string(m)['checkpoint_file'] for m in ensemble]
        python_cmd.append('--checkpoint_paths')
        python_cmd.extend([os.path.join('/checkpoints/',cp) for cp in checkpoint_paths]) # We're going to mount the checkpoint folder below

        cmd = [
            'nvidia-docker','run',
            '-v','{}:/input_images'.format(os.path.abspath(input_dir)),
            '-v','{}:/output_data'.format(os.path.abspath(output_dir)),
            '-v','{}:/code'.format(os.path.abspath(os.getcwd())),
            '-v', '{}:/checkpoints'.format(os.path.abspath(CHECKPOINT_DIR)),
            '-w','/code',
            'rwightman/pytorch-extra'
        ]

        cmd.extend(python_cmd)

        subprocess.call(cmd)

        subprocess.call(['python', 'evaluate_attacks_and_defenses.py'])

# Run all base defenses against existing attacks
def complete_remaining():
    attacks = os.listdir(os.path.join(main_dir, 'attacks'))
    for a in attacks:
        for d in all_models:
            run_base_defense_experiment([d], [1.0], a, False)
    targeted_attacks = os.listdir(os.path.join(main_dir, 'targeted_attacks'))
    for a in targeted_attacks:
        for d in all_models:
            run_base_defense_experiment([d], [1.0], a, True)

all_models = [
    'inception_v3',
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

models_exclude_for_attack = ['DPN107Extra']
all_models_for_attacks = [m for m in all_models if m not in models_exclude_for_attack]

run_cw_inspired_experiment(['adv_inception_resnet_v2', 'inception_v3'],[1.0, 1.0],targeted=True,lr=0.08,no_augmentation=True)
run_cw_inspired_experiment(['adv_inception_resnet_v2', 'inception_v3', 'resnet34'],[1.0, 1.0, 0.885],targeted=False,lr=0.16,n_iter=80)

# Cleanup!
complete_remaining()
"""
for t in [False]:
    for m in all_models_for_attacks:
        run_cw_inspired_experiment(
            [m],
            [1.0],
            targeted = t)
        complete_remaining()
"""