import os
import sys
import argparse
from string import Template
import yaml
import shutil
import json
import subprocess

path = os.path.abspath('../python/')
sys.path.insert(0, path)
from models.model_configs import config_from_string

with open('../python/local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
CHECKPOINT_DIR = local_config['checkpoints_dir']
DEPLOYMENT_DIR = local_config['deployment_dir']

parser = argparse.ArgumentParser(description='No description')
parser.add_argument('--name', type=str, help='Name of json file')
parser.add_argument('--type', type=str, help='targeted_attack, attack, or defense')

def deploy_attack(cfg, dont_tar=False):
    run_template_path = os.path.join('../python/attacks', 'run_attack.sh.template')
    metadata_template_path = os.path.join('../python/attacks', 'metadata.json.template')

    name = cfg['name']
    attack_type = cfg['attack_type']
    run_cmd = cfg['run_cmd']

    if 'runargs' in cfg:
        runargs = cfg['runargs']
    else:
        runargs = []

    deployment_path = os.path.join(DEPLOYMENT_DIR, 'targeted_attacks' if attack_type == 'targeted_attack' else 'attacks', name)
    if os.path.exists(deployment_path):
        shutil.rmtree(deployment_path)
    os.makedirs(deployment_path)

    if 'ensemble' in cfg:
        ensemble = cfg['ensemble']
        ensemble_weights = cfg['ensemble_weights']

        checkpoint_paths = [config_from_string(m)['checkpoint_file'] for m in ensemble]
        checkpoints_to_copy = [os.path.join(CHECKPOINT_DIR, cp) for cp in checkpoint_paths]
        for cp_src, cp_dst in zip(checkpoints_to_copy, checkpoint_paths):
            shutil.copy(cp_src, os.path.join(deployment_path, cp_dst))

        runargs.append('--ensemble')
        runargs.extend(ensemble)
        runargs.append('--ensemble_weights')
        runargs.extend(ensemble_weights)
        runargs.append('--checkpoint_paths')
        runargs.extend(checkpoint_paths)

    if 'npy_file' in cfg:
        npy_file = cfg['npy_file']
        runargs.extend(['--npy_file', os.path.join('python',npy_file)])

    if 'npy_files' in cfg:
        npy_files = cfg['npy_files']
        npy_files_arg = ['--npy_files']
        npy_files_arg.extend([os.path.join('python', nf) for nf in npy_files])
        runargs.extend(npy_files_arg)

    shutil.copytree('../python', os.path.join(deployment_path, 'python'))

    with open(run_template_path, 'r') as run_template_file:
        run_template = Template(run_template_file.read())
        run_sh = run_template.substitute({'run_cmd': run_cmd, 'run_args': ' '.join(runargs)})
        run_sh_path = os.path.join(deployment_path, 'run_attack.sh')

        with open(run_sh_path, 'w') as run_sh_file:
            run_sh_file.write(run_sh)

        # Otherwise submission will be invalid without execute permission
        # Though they claim to have fixed this on their end, was necessary
        # in rounds 1 and 2.
        os.chmod(run_sh_path, 777)

    with open(metadata_template_path, 'r') as metadata_template_file:
        metadata_template = Template(metadata_template_file.read())
        metadata_json = metadata_template.substitute({'attack_type': attack_type})

        with open(os.path.join(deployment_path, 'metadata.json'), 'w') as metadata_json_file:
            metadata_json_file.write(metadata_json)

    if not dont_tar:
        tar_filepath = os.path.join(deployment_path, '../{}.tar.gz'.format(name))
        subprocess.call(['tar', '-czvf', tar_filepath, '-C', deployment_path, '.', '--strip-components=1'])

        subprocess.call(['python',
                         'validation_tool/validate_submission.py',
                         '--submission_filename', os.path.abspath(tar_filepath),
                         '--submission_type', attack_type,
                         '--use_gpu'])

    return deployment_path

def deploy_defense(cfg, dont_tar=False):
    run_template_path = os.path.join('../python/defenses', 'run_defense.sh.template')
    metadata_template_path = os.path.join('../python/defenses', 'metadata.json.template')

    name = cfg['name']
    run_cmd = cfg['run_cmd']

    if 'runargs' in cfg:
        runargs = cfg['runargs']
    else:
        runargs = []

    deployment_path = os.path.join(DEPLOYMENT_DIR, 'defenses', name)
    if os.path.exists(deployment_path):
        shutil.rmtree(deployment_path)
    os.makedirs(deployment_path)

    if 'ensemble' in cfg:
        ensemble = cfg['ensemble']
        ensemble_weights = cfg['ensemble_weights']

        checkpoint_paths = [config_from_string(m)['checkpoint_file'] for m in ensemble]
        checkpoints_to_copy = [os.path.join(CHECKPOINT_DIR, cp) for cp in checkpoint_paths]
        for cp_src, cp_dst in zip(checkpoints_to_copy, checkpoint_paths):
            shutil.copy(cp_src, os.path.join(deployment_path, cp_dst))

        runargs.append('--ensemble')
        runargs.extend(ensemble)
        runargs.append('--ensemble_weights')
        runargs.extend(ensemble_weights)
        runargs.append('--checkpoint_paths')
        runargs.extend(checkpoint_paths)

    if 'npy_file' in cfg:
        runargs.extend(['--npy_file', os.path.join('python', cfg['npy_file'])])

    if 'npy_files' in cfg:
        npy_files = [os.path.join('python', f) for f in cfg['npy_files']]
        runargs.append('--npy_files')
        runargs.extend(npy_files)

    shutil.copytree('../python', os.path.join(deployment_path, 'python'))

    with open(run_template_path, 'r') as run_template_file:
        run_template = Template(run_template_file.read())
        run_sh = run_template.substitute({'run_cmd': run_cmd, 'run_args': ' '.join(runargs)})
        run_sh_path = os.path.join(deployment_path, 'run_defense.sh')

        with open(run_sh_path, 'w') as run_sh_file:
            run_sh_file.write(run_sh)

        # Otherwise submission will be invalid without execute permission
        # Though they claim to have fixed this on their end, was necessary
        # in rounds 1 and 2.
        os.chmod(run_sh_path, 777)

    with open(metadata_template_path, 'r') as metadata_template_file:
        metadata_template = Template(metadata_template_file.read())
        metadata_json = metadata_template.substitute({})

        with open(os.path.join(deployment_path, 'metadata.json'), 'w') as metadata_json_file:
            metadata_json_file.write(metadata_json)

    if not dont_tar:
        tar_filepath = os.path.join(deployment_path, '../{}.tar.gz'.format(name))
        subprocess.call(['tar', '-czvf', tar_filepath, '-C', deployment_path, '.', '--strip-components=1'])

        subprocess.call(['python',
                         'validation_tool/validate_submission.py',
                         '--submission_filename', os.path.abspath(tar_filepath),
                         '--submission_type', 'defense',
                         '--use_gpu'])

    return deployment_path


def main():
    args = parser.parse_args()

    with open(os.path.join(args.type, '{}.json'.format(args.name)), 'r') as json_file:
        json_data = json.load(json_file)

    if args.type in ['attack', 'targeted_attack']:
        deploy_attack(json_data)
    elif args.type == 'defense':
        deploy_defense(json_data)
    else:
        raise ValueError('Only attack, targeted_attack, or defense are valid.')

if __name__ == '__main__':
    main()
