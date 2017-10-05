from .torchvision import Resnet18
from ..load_checkpoint import load_checkpoint
from ..transformed_model import TransformedModel
#from ...processing import Normalize

def create_model(
        model_name,
        num_classes=10, # Default CIFAR-10
        output_fn='log_softmax',
        checkpoint_path=None,
        **kwargs):

    if model_name == 'resnet18':
        model = Resnet18(num_classes)
    else:
        raise ValueError('Invalid model_name: {}'.format(model_name))

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    if num_classes == 10:
        normalizer = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif num_classes == 100:
        normalizer = Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    else:
        raise ValueError('Invalid number of CIFAR classes: {}'.format(num_classes))

    model = TransformedModel(
        model=model,
        input_size=None,
        normalizer=normalizer,
        output_fn=output_fn
    )

    return model

