from .ensemble import Ensemble
from .model_factory import create_model_from_cfg


def create_ensemble(model_configs, ensembling_weights, checkpoint_paths=[], dataset='imagenet'):
    models = []
    for i, mc in enumerate(model_configs):
        checkpoint_path = checkpoint_paths[i] if checkpoint_paths else ''
        models.append(create_model_from_cfg(mc, checkpoint_path, dataset))

    return Ensemble(models, ensembling_weights)
