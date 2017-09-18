import os
import sys
import yaml
import subprocess

sys.path.insert(0, os.path.abspath('../python'))
from deployment import deploy
from experiment.validate_attack_inf_norm import validate

with open('../python/local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
main_dir = local_config['results_dir']
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

IMAGES_DIR = local_config['images_dir']


class Experiment:
    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError()

    def get_cfg(self):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()


class AttackExperiment(Experiment):
    def __init__(self, max_epsilon, targeted):
        super(AttackExperiment, self).__init__()
        self.max_epsilon = max_epsilon
        self.targeted = targeted

    def run(self):
        experiment_name = self.get_name()
        cfg = self.get_cfg()

        output_dir = os.path.join(main_dir, '{}s'.format(cfg['attack_type']), experiment_name)
        if not os.path.exists(output_dir):
            print('Running experiment: {}'.format(experiment_name))

            deployment_path = deploy.deploy_attack(cfg, dont_tar=True)

            internal_cmd = [
                'sh', 'run_attack.sh',
                '../input_images',
                '../output_images',
                str(self.max_epsilon),
                ]

            docker_cmd = [
                'nvidia-docker', 'run',
                '-v', '{}:/input_images'.format(os.path.abspath(IMAGES_DIR)),
                '-v', '{}:/output_images'.format(os.path.abspath(output_dir)),
                '-v', '{}:/code'.format(os.path.abspath(deployment_path)),
                '-w', '/code',
                'rwightman/pytorch-extra'
            ]

            docker_cmd.extend(internal_cmd)

            subprocess.call(docker_cmd)

            validate(experiment_name, self.targeted)

    def get_cfg(self):
        return super(AttackExperiment, self).get_cfg()

    def get_name(self):
        return super(AttackExperiment, self).get_name()


class DefenseExperiment(Experiment):
    def __init__(self, attack_type, attack_name):
        super(DefenseExperiment, self).__init__()
        self.attack_type = attack_type
        self.attack_name = attack_name

    def run(self):
        experiment_name = self.get_name()
        cfg = self.get_cfg()

        output_path = os.path.join(main_dir, 'defenses', experiment_name, '{}s'.format(self.attack_type))
        output_file = '{}.csv'.format(self.attack_name)
        if not os.path.exists(os.path.join(output_path, output_file)):
            print('Running experiment: {}'.format(experiment_name))

            deployment_path = deploy.deploy_defense(cfg, dont_tar=True)

            internal_cmd = [
                'sh', 'run_defense.sh',
                '../input_images',
                '../output_data/{}'.format(output_file)
                ]

            docker_cmd = [
                'nvidia-docker', 'run',
                '-v', '{}:/input_images'.format(os.path.abspath(os.path.join(main_dir, '{}s'.format(self.attack_type), self.attack_name))),
                '-v', '{}:/output_data'.format(os.path.abspath(output_path)),
                '-v', '{}:/code'.format(os.path.abspath(deployment_path)),
                '-w', '/code',
                'rwightman/pytorch-extra'
            ]

            docker_cmd.extend(internal_cmd)

            subprocess.call(docker_cmd)

            subprocess.call(['python', '../experiment/evaluate_attacks_and_defenses.py'])

    def get_cfg(self):
        return super(DefenseExperiment, self).get_cfg()

    def get_name(self):
        return super(DefenseExperiment, self).get_name()
