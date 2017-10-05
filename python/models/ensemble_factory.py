from .ensemble import Ensemble
from .model_factory import create_model
from .cifar.model_factory import create_model as create_cifar_model
from .mnist.model_factory import create_model as create_mnist_model

def create_ensemble(model_configs, ensembling_weights, checkpoint_paths=[], dataset='imagenet'):
    models = []
    for i, mc in enumerate(model_configs):
        if 'kwargs' not in mc:
            mc['kwargs'] = {}

        if dataset == 'imagenet':
            models.append(create_model(
                model_name=mc['model_name'],
                num_classes=mc['num_classes'],
                input_size=mc['input_size'],
                normalizer=mc['normalizer'],
                output_fn=mc['output_fn'],
                drop_first_class=mc['drop_first_class'],
                checkpoint_path=checkpoint_paths[i] if checkpoint_paths else mc['checkpoint_file'],
                **mc['kwargs']
            ))
        elif dataset == 'cifar':
            models.append(create_cifar_model(
                model_name=mc['model_name'],
                num_classes=mc['num_classes'],
                checkpoint_path=checkpoint_paths[i] if checkpoint_paths else mc['checkpoint_file'],
                **mc['kwargs']
            ))
        elif dataset == 'mnist':
            models.append(create_mnist_model(
                model_name=mc['model_name'],
                checkpoint_path=checkpoint_paths[i] if checkpoint_paths else mc['checkpoint_file'],
                **mc['kwargs']
            ))

    return Ensemble(models, ensembling_weights)
