import os
import sys
path = os.path.abspath('../python/')
sys.path.insert(0, path)

import argparse
from models.model_configs import config_from_string
from string import Template
import yaml
import shutil
import json

with open('../python/local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
CHECKPOINT_DIR = local_config['checkpoints_dir']
DEPLOYMENT_DIR = local_config['deployment_dir']

parser = argparse.ArgumentParser(description='No description')
parser.add_argument('--name', type=str, help='Name of json file')
parser.add_argument('--type', type=str, help='targeted_attack, attack, or defense')
parser.add_argument('--no_extra_args', action='store_true', default=False, help='Use this for universal.')


def deploy_attack(cfg, no_extra_args=False):
    run_template_path = os.path.join('../python/attacks','run_attack.sh.template')
    metadata_template_path = os.path.join('../python/attacks','metadata.json.template')

    name = cfg['name']
    attack_type = cfg['attack_type']
    ensemble = cfg['ensemble']
    ensemble_weights = cfg['ensemble_weights']
    run_cmd = cfg['run_cmd']
    runargs = cfg['base_runargs']

    checkpoint_paths = [config_from_string(m)['checkpoint_file'] for m in ensemble]

    deployment_path = os.path.join(DEPLOYMENT_DIR, 'targeted_attacks' if attack_type == 'targeted_attack' else 'attacks', name)

    if not no_extra_args:
        runargs.append('--ensemble')
        runargs.extend(ensemble)
        runargs.append('--ensemble_weights')
        runargs.extend(ensemble_weights)
        runargs.append('--checkpoint_paths')
        runargs.extend(checkpoint_paths)

    if not os.path.exists(deployment_path):
        os.makedirs(deployment_path)

    checkpoints_to_copy = [os.path.join(CHECKPOINT_DIR,cp) for cp in checkpoint_paths]
    for cp_src, cp_dst in zip(checkpoints_to_copy, checkpoint_paths):
        shutil.copy(cp_src, os.path.join(deployment_path,cp_dst))

    shutil.copytree('../python',os.path.join(deployment_path,'python'))

    with open(run_template_path, 'r') as run_template_file:
        run_template = Template(run_template_file.read())
        run_sh = run_template.substitute({'run_cmd': run_cmd, 'run_args': ' '.join(runargs)})

        with open(os.path.join(deployment_path,'run_attack.sh'), 'w') as run_sh_file:
            run_sh_file.write(run_sh)

        os.chmod(os.path.join(deployment_path,'run_attack.sh'), 777)

    with open(metadata_template_path, 'r') as metadata_template_file:
        metadata_template = Template(metadata_template_file.read())
        metadata_json = metadata_template.substitute({'attack_type': attack_type})

        with open(os.path.join(deployment_path,'metadata.json'), 'w') as metadata_json_file:
            metadata_json_file.write(metadata_json)

def main():
    args = parser.parse_args()

    with open(os.path.join(args.type, '{}.json'.format(args.name)), 'r') as json_file:
        json_data = json.load(json_file)

    if args.type in ['attack', 'targeted_attack']:
        deploy_attack(json_data, no_extra_args=args.no_extra_args)
    elif args.type == 'defense':
        raise NotImplementedError()
    else:
        raise ValueError('Only attack, targeted_attack, or defese is valid.')

if __name__=='__main__':
    main()
