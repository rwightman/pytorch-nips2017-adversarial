from experiment.experiment import AttackExperiment


class SingleUniversalExperiment(AttackExperiment):
    RUN_CMD = 'run_universal_perturbation'

    def __init__(
        self,
        npy_file,
        max_epsilon=16
    ):
        super(SingleUniversalExperiment, self).__init__(max_epsilon, False)

        self.npy_file = npy_file

    def get_name(self):
        experiment_name = 'universal_{}'.format(self.npy_file)

        return experiment_name

    def get_cfg(self):
        cfg = {}
        cfg['name'] = self.get_name()
        cfg['attack_type'] = 'attack'
        cfg['run_cmd'] = SingleUniversalExperiment.RUN_CMD
        cfg['npy_file'] = self.npy_file
        cfg['runargs'] = []

        return cfg