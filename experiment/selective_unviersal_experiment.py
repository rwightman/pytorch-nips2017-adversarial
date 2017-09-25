from experiment import AttackExperiment


class SelectiveUniversalExperiment(AttackExperiment):
    RUN_CMD = 'run_selective_universal'

    def __init__(
        self,
        npy_files,
        ensemble,
        ensemble_weights,
        max_epsilon=16,
        try_mirrors = False
    ):
        super(SelectiveUniversalExperiment, self).__init__(max_epsilon, False)

        self.npy_files = npy_files
        self.ensemble = ensemble
        self.ensemble_weights = ensemble_weights
        self.try_mirrors = try_mirrors

    def get_name(self):
        npy_two_letters = [x[0:2] for x in self.npy_files]
        experiment_name = 'eps{}_selective_{}_{}'.format(self.max_epsilon, (''.join(npy_two_letters)).replace('.npy',''),''.join(self.ensemble))
        for model, weight in zip(self.ensemble, self.ensemble_weights):
            experiment_name = '{}{}{}'.format(experiment_name, model, weight)
        if self.try_mirrors:
            experiment_name = '{}_mirror'.format(experiment_name)

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
        if self.try_mirrors:
            cfg['runargs'].append('--try_mirrors')

        return cfg