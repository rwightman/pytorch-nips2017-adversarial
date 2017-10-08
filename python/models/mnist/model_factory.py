from .networks import MadryNet, PytorchExampleNet
from .ens_adv_training import ModelA, ModelB, ModelC, ModelD
from models.load_checkpoint import load_checkpoint
from models.transformed_model import TransformedModel
from processing import Normalize

def create_model(
        model_name,
        output_fn='log_softmax',
        checkpoint_path=None,
        **kwargs):

    if model_name == 'madry':
        model = MadryNet()
    elif model_name == 'pytorch-example':
        model = PytorchExampleNet()
    elif model_name == 'modela':
        model = ModelA()
    elif model_name == 'modelb':
        model = ModelB()
    elif model_name == 'modelc':
        model = ModelC()
    elif model_name == 'modeld':
        model = ModelD()
    else:
        raise ValueError('Invalid model_name: {}'.format(model_name))

    if output_fn:
        model = TransformedModel(
            model=model,
            input_size=28,
            normalizer=Normalize((0.1307,), (0.3081,)),
            output_fn=output_fn
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model

