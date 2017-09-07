from .ensemble import Ensemble
from .model_factory import create_model


def create_ensemble(model_configs, ensembling_weights, checkpoint_paths=[]):
    models = []
    for i, mc in enumerate(model_configs):
        models.append(create_model(
            model_name=mc['model_name'],
            num_classes=mc['num_classes'],
            input_size=mc['input_size'],
            normalizer=mc['normalizer'],
            output_fn=mc['output_fn'],
            drop_first_class=mc['drop_first_class'],
            checkpoint_path=checkpoint_paths[i] if checkpoint_paths else mc['checkpoint_file']
        ))

    return Ensemble(models, ensembling_weights)
