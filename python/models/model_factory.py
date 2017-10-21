import torchvision.models
from .resnext101_32x4d import resnext101_32x4d
from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .my_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fbresnet200 import fbresnet200
from .dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from .transformed_model import TransformedModel
from .load_checkpoint import load_checkpoint

from .cifar.model_factory import create_model as create_cifar_model
from .mnist.model_factory import create_model as create_mnist_model

def create_model(
        model_name='resnet50',
        pretrained=False,
        num_classes=1000,
        input_size=0,
        normalizer='',
        drop_first_class=False,
        output_fn='',
        checkpoint_path='',
        **kwargs):

    if 'test_time_pool' in kwargs:
        test_time_pool = kwargs.pop('test_time_pool')
    else:
        test_time_pool = True

    if model_name == 'dpn68':
        model = dpn68(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn68b':
        model = dpn68b(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn92':
        model = dpn92(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn98':
        model = dpn98(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn131':
        model = dpn131(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn107':
        model = dpn107(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet18-torchvision':
        model = torchvision.models.resnet18(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet34-torchvision':
        model = torchvision.models.resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet50-torchvision':
        model = torchvision.models.resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101-torchvision':
        model = torchvision.models.resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet152-torchvision':
        model = torchvision.models.resnet152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121-torchvision':
        model = torchvision.models.densenet121(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet161':
        model = densenet161(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet161-torchvision':
        model = torchvision.models.densenet161(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet169':
        model = densenet169(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet169-torchvision':
        model = torchvision.models.densenet169(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet201':
        model = densenet201(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet201-torchvision':
        model = torchvision.models.densenet201(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_v3':
        model = torchvision.models.inception_v3(
            num_classes=num_classes, pretrained=pretrained, transform_input=False, aux_logits=False)
    elif model_name == 'inception_resnet_v2':
        model = inception_resnet_v2(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_v4':
        model = inception_v4(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'resnext101_32x4d':
        model = resnext101_32x4d(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'wrn50':
        model = wrn50_2(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'fbresnet200':
        model = fbresnet200(num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained=pretrained)
    elif model_name == 'squeezenet1_1':
        model = torchvision.models.squeezenet1_1(pretrained=pretrained)
    elif model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=pretrained)
    else:
        assert False and "Invalid model"

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    if input_size or normalizer or drop_first_class or output_fn:
        model = TransformedModel(
            model=model,
            input_size=input_size,
            normalizer=normalizer,
            output_fn=output_fn,
            drop_first_class=drop_first_class,
        )

    return model


def create_model_from_cfg(mc, checkpoint_path='', dataset='imagenet'):
    if 'kwargs' not in mc:
        mc['kwargs'] = {}

    if 'dataset' in mc:
        dataset = mc.pop('dataset')

    if dataset == 'imagenet':
        model = create_model(
            model_name=mc['model_name'],
            num_classes=mc['num_classes'],
            input_size=mc['input_size'],
            normalizer=mc['normalizer'],
            output_fn=mc['output_fn'],
            drop_first_class=mc['drop_first_class'],
            checkpoint_path=checkpoint_path if checkpoint_path else mc['checkpoint_file'],
            **mc['kwargs']
        )
    elif dataset == 'cifar':
        model = create_cifar_model(
            model_name=mc['model_name'],
            num_classes=mc['num_classes'],
            checkpoint_path=checkpoint_path if checkpoint_path else mc['checkpoint_file'],
            **mc['kwargs']
        )
    elif dataset == 'mnist':
        model = create_mnist_model(
            model_name=mc['model_name'],
            checkpoint_path=checkpoint_path if checkpoint_path else mc['checkpoint_file'],
            **mc['kwargs']
        )

    return model


