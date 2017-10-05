from .networks import MadryNet, PytorchExampleNet
from ..load_checkpoint import load_checkpoint
from ..transformed_model import TransformedModel
#from ...processing import Normalize

def create_model(
        model_name,
        output_fn='log_softmax',
        checkpoint_path=None,
        **kwargs):

    if model_name == 'madry':
        model = MadryNet()
    elif model_name == 'pytorch-example':
        model = PytorchExampleNet()
    else:
        raise ValueError('Invalid model_name: {}'.format(model_name))

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    if output_fn:
        model = TransformedModel(
            model=model,
            input_size=None,
            normalizer=Normalize((0.1307,), (0.3081,)),
            output_fn=output_fn
        )

    return model

