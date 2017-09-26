from experiment import AttackExperiment


class SmartUniversalExperiment(AttackExperiment):
    RUN_CMD = 'run_smart_universal'

    def __init__(
        self,
        npy_files,
        ensemble,
        ensemble_weights,
        max_epsilon=16,
        try_mirrors = False,
        no_augmentation=False,
        lr=0.02,
    ):
        super(SmartUniversalExperiment, self).__init__(max_epsilon, False)

        self.npy_files = npy_files
        self.ensemble = ensemble
        self.ensemble_weights = ensemble_weights
        self.try_mirrors = try_mirrors
        self.no_augmentation = no_augmentation
        self.lr = lr

    def get_name(self):
        npy_two_letters = [x[0:2] for x in self.npy_files]
        experiment_name = 'eps{}_smart_{}_{}'.format(self.max_epsilon, (''.join(npy_two_letters)).replace('.npy',''),''.join(self.ensemble))
        for model, weight in zip(self.ensemble, self.ensemble_weights):
            experiment_name = '{}{}{}'.format(experiment_name, model, weight)
        if self.try_mirrors:
            experiment_name = '{}_mirror'.format(experiment_name)
        if self.no_augmentation:
            experiment_name = '{}_noaug'.format(experiment_name)
        if self.lr != 0.02:
            experiment_name = '{}_lr{}'.format(experiment_name, self.lr)

        return experiment_name

    def get_cfg(self):
        cfg = {}
        cfg['name'] = self.get_name()
        cfg['attack_type'] = 'attack'
        cfg['run_cmd'] = SmartUniversalExperiment.RUN_CMD
        cfg['npy_files'] = self.npy_files
        cfg['ensemble'] = self.ensemble
        cfg['ensemble_weights'] = [str(w) for w in self.ensemble_weights]
        cfg['runargs'] = []
        if self.try_mirrors:
            cfg['runargs'].append('--try_mirrors')
        if self.no_augmentation:
            cfg['runargs'].append('--no_augmentation')
        if self.lr != 0.02:
            cfg['runargs'].extend(['--lr', str(self.lr)])
        cfg['runargs'].extend(['--time_limit_per_100',str(175.0)])
        cfg['runargs'].extend(['--batch_size', '8'])

        return cfg