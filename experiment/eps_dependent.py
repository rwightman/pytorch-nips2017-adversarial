from experiment import AttackExperiment


class EpsilonDependentExperiment(AttackExperiment):
    def __init__(
        self,
        epsilon_dict,
        targeted,
        name,
        max_epsilon=16,
    ):
        super(EpsilonDependentExperiment, self).__init__(max_epsilon, targeted)

        self.epsilon_dict = epsilon_dict
        self.name = name

    def get_name(self):
        return self.name

    def get_cfg(self):
        cfg = {}
        cfg['name'] = self.get_name()
        cfg['attack_type'] = 'targeted_attack' if self.targeted else 'attack'
        cfg['epsilon_dict'] = {}
        for epsilon, experiment in self.epsilon_dict.items():
            cfg['epsilon_dict'][str(epsilon)] = experiment.get_cfg()

        return cfg