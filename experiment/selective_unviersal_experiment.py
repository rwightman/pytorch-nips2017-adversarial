from experiment.experiment import AttackExperiment


class SelectiveUniversalExperiment(AttackExperiment):
    RUN_CMD = 'run_selective_universal'

    def __init__(
        self,
        npy_files,
        ensemble,
        ensemble_weights,
        max_epsilon=16
    ):
        super(SelectiveUniversalExperiment, self).__init__(max_epsilon, False)

        self.npy_files = npy_files
        self.ensemble = ensemble
        self.ensemble_weights = ensemble_weights

    def get_name(self):
        experiment_name = 'selective_{}_{}'.format((''.join(self.npy_files)).replace('.npy',''),''.join(self.ensemble))

        return experiment_name

    def get_cfg(self):
        cfg = {}
        cfg['name'] = self.get_name()
        cfg['attack_type'] = 'attack'
        cfg['run_cmd'] = SelectiveUniversalExperiment.RUN_CMD
        cfg['npy_files'] = self.npy_files
        cfg['ensemble'] = self.ensemble
        cfg['ensemble_weights'] = [str(w) for w in self.ensemble_weights]
        cfg['runargs'] = []

        return cfg