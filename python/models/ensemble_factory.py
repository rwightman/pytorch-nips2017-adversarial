from .ensemble import  Ensemble
from .model_factory import create_model


def create_ensemble(model_configs, ensembling_weights):
    models = []
    for mc in model_configs:
        models.append(create_model(
            model_name=mc['model_name'],
            pretrained=mc['pretrained'],
            num_classes=mc['num_classes'],
            normalize_inputs=mc['normalize_inputs'],
            resize_inputs=mc['resize_inputs'],
            input_size=mc['input_size'],
            standardize_outputs=mc['standardize_outputs'],
            drop_first_class=mc['drop_first_class']
        ))

    return Ensemble(models,ensembling_weights)