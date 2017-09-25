from experiment import AttackExperiment
from models.model_configs import config_from_string

class CWInspiredExperiment(AttackExperiment):
    RUN_CMD = 'run_cw_inspired'

    def __init__(
        self,
        ensemble,
        ensemble_weights,
        targeted,
        max_epsilon=16,
        no_augmentation=False,
        no_augmentation_blurring=False,
        n_iter=100,
        lr=0.02,
        target_nth_highest=6,
        gaus_blur_prob=0.5,
        gaus_blur_size=5,
        gaus_blur_sigma=3.0,
        brightness_contrast=False,
        saturation=False,
        prob_dont_augment=0.0,
        initial_w_matrix=None
    ):
        super(CWInspiredExperiment, self).__init__(max_epsilon, targeted)

        self.ensemble = ensemble
        self.ensemble_weights = ensemble_weights
        self.no_augmentation = no_augmentation
        self.no_augmentation_blurring = no_augmentation_blurring
        self.n_iter = n_iter
        self.lr = lr
        self.target_nth_highest = target_nth_highest
        self.gaus_blur_prob = gaus_blur_prob
        self.gaus_blur_size = gaus_blur_size
        self.gaus_blur_sigma = gaus_blur_sigma
        self.brightness_contrast = brightness_contrast
        self.saturation = saturation
        self.prob_dont_augment = prob_dont_augment
        self.initial_w_matrix = initial_w_matrix

    def get_name(self):
        experiment_name = 'eps{}_cw_inspired_'.format(self.max_epsilon)
        for model, weight in zip(self.ensemble, self.ensemble_weights):
            experiment_name = '{}{}{}'.format(experiment_name, model, weight)

        if self.no_augmentation:
            experiment_name = '{}_noaug'.format(experiment_name)
        if self.no_augmentation_blurring:
            experiment_name = '{}_noblur'.format(experiment_name)
        if self.n_iter != 100:
            experiment_name = '{}_{}iter'.format(experiment_name, self.n_iter)
        if self.lr != 0.02:
            experiment_name = '{}_lr{}'.format(experiment_name, self.lr)
        if self.target_nth_highest != 6:
            experiment_name = '{}_trg{}'.format(experiment_name, self.target_nth_highest)

        # Gaus Blur Experiment Variables
        if self.gaus_blur_prob != 0.5:
            experiment_name = '{}_gbp{}'.format(experiment_name, self.gaus_blur_prob)
        if self.gaus_blur_size != 5:
            experiment_name = '{}_gbsiz{}'.format(experiment_name, self.gaus_blur_size)
        if self.gaus_blur_sigma != 3.0:
            experiment_name = '{}_gbsig{}'.format(experiment_name, self.gaus_blur_sigma)

        if self.brightness_contrast:
            experiment_name = '{}_bri'.format(experiment_name)
        if self.saturation:
            experiment_name = '{}_sat'.format(experiment_name)
        if self.prob_dont_augment != 0.0:
            experiment_name = '{}_dontaug{}'.format(experiment_name, self.prob_dont_augment)

        if self.initial_w_matrix is not None:
            experiment_name = '{}_w{}'.format(experiment_name, self.initial_w_matrix)

        return experiment_name

    def get_cfg(self):
        cfg = {}
        cfg['name'] = self.get_name()
        cfg['attack_type'] = 'targeted_attack' if self.targeted else 'attack'
        cfg['ensemble'] = self.ensemble
        cfg['ensemble_weights'] = [str(w) for w in self.ensemble_weights]
        cfg['run_cmd'] = CWInspiredExperiment.RUN_CMD

        runargs = []

        checkpoint_paths = [config_from_string(m)['checkpoint_file'] for m in self.ensemble]
        runargs.append('--checkpoint_paths')
        runargs.extend(checkpoint_paths)

        if self.targeted:
            runargs.append('--targeted')
        if self.no_augmentation:
            runargs.append('--no_augmentation')
        if self.no_augmentation_blurring:
            runargs.append('--no_augmentation_blurring')
        if self.n_iter != 100:
            runargs.extend(['--n_iter', str(self.n_iter)])
        if self.lr != 0.02:
            runargs.extend(['--lr', str(self.lr)])
        if self.target_nth_highest != 6:
            runargs.extend(['--target_nth_highest', str(self.target_nth_highest)])
            runargs.extend(['--gaus_blur_prob', str(self.gaus_blur_prob),
                            '--gaus_blur_size', str(self.gaus_blur_size),
                            '--gaus_blur_sigma', str(self.gaus_blur_sigma)])
        if self.brightness_contrast:
            runargs.append('--brightness_contrast')
        if self.saturation:
            runargs.append('--saturation')
        if self.prob_dont_augment != 0.0:
            runargs.extend(['--prob_dont_augment', str(self.prob_dont_augment)])
        if self.initial_w_matrix is not None:
            runargs.extend(['--initial_w_matrix', self.initial_w_matrix])

        runargs.extend(['--batch_size', '8'])

        cfg['runargs'] = runargs

        return cfg