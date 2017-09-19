from experiment import DefenseExperiment
from models.model_configs import config_from_string


class BaseExperiment(DefenseExperiment):
    RUN_CMD = 'run_base'

    def __init__(
        self,
        ensemble,
        ensemble_weights,
        attack_name,
        attack_type
    ):
        super(BaseExperiment, self).__init__(attack_type, attack_name)
        self.ensemble = ensemble
        self.ensemble_weights = ensemble_weights


    def get_name(self):
        experiment_name = 'base_{}'.format(''.join(sorted(self.ensemble)))
        return experiment_name

    def get_cfg(self):
        cfg = {}
        cfg['name'] = self.get_name()
        cfg['ensemble'] = self.ensemble
        cfg['ensemble_weights'] = [str(w) for w in self.ensemble_weights]
        cfg['run_cmd'] = BaseExperiment.RUN_CMD

        runargs = []

        checkpoint_paths = [config_from_string(m)['checkpoint_file'] for m in self.ensemble]
        runargs.append('--checkpoint_paths')
        runargs.extend(checkpoint_paths)

        runargs.extend(['--batch-size', '8'])

        cfg['runargs'] = runargs

        return cfg
