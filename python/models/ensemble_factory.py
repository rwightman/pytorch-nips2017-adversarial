from .ensemble import Ensemble
from .model_factory import create_model_from_cfg


def create_ensemble(model_configs, ensembling_weights, checkpoint_paths=[], mean_method='arithmetic'):
    models = []
    for i, mc in enumerate(model_configs):
        checkpoint_path = checkpoint_paths[i] if checkpoint_paths else ''
        models.append(create_model_from_cfg(mc, checkpoint_path))

    return Ensemble(models, ensembling_weights, mean_method=mean_method)
