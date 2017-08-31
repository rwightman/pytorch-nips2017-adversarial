import torchvision.models
from .resnext101_32x4d import resnext101_32x4d
from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .my_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fbresnet200 import fbresnet200
from .dpn import dpn68, dpn92, dpn98, dpn131, dpn107
from .normalized_model import NormalizedModel
from .self_resizing_model import SelfResizingModel
from .standardized_output_model import StandardizedOutputModel
from normalizations.normalizer_factory import get_normalizer

model_name_normalizer_name_mapping = {
    'dpn68' : 'dualpathnet',
    'dpn92' : 'dualpathnet',
    'dpn131' : 'dualpathnet',
    'dnp107' : 'dualpathnet',
    'resnet18' : 'torchvision',
    'resnet34' : 'torchvision',
    'resnet50' : 'torchvision',
    'resnet101' : 'torchvision',
    'resnet152' : 'torchvision',
    'resnet18-torchvision': 'torchvision',
    'resnet34-torchvision': 'torchvision',
    'resnet50-torchvision': 'torchvision',
    'resnet101-torchvision': 'torchvision',
    'resnet152-torchvision': 'torchvision',
    'densenet121' : 'torchvision',
    'densenet161' : 'torchvision',
    'densenet169' : 'torchvision',
    'densenet201' : 'torchvision',
    'densenet121-torchvision': 'torchvision',
    'densenet161-torchvision': 'torchvision',
    'densenet169-torchvision': 'torchvision',
    'densenet201-torchvision': 'torchvision',
    'squeezenet1_0': 'torchvision',
    'squeezenet1_1': 'torchvision',
    'alexnet' : 'torchvision',
    'inception_v3' : 'le',
    'inception_resnet_v2' : 'le',
    'inception_v4' : 'le',
    'resnext101_32x4d' : None,
    'wrn50' : None,
    'fbresnet200' : None
}


def create_model(
        model_name='resnet50',
        pretrained=True,
        num_classes=1000,
        normalize_inputs=False,
        resize_inputs=False,
        standardize_outputs=False,
        **kwargs):

    if 'test_time_pool' in kwargs:
        test_time_pool = kwargs.pop('test_time_pool')
    else:
        test_time_pool = True

    if 'drop_first_class' in kwargs:
        drop_first_class = kwargs.pop('drop_first_class')
    if 'input_size' in kwargs:
        input_size = kwargs.pop('input_size')

    if model_name == 'dpn68':
        model = dpn68(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool, **kwargs)
    elif model_name == 'dpn92':
        model = dpn92(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool, **kwargs)
    elif model_name == 'dpn98':
        model = dpn98(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool, **kwargs)
    elif model_name == 'dpn131':
        model = dpn131(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool, **kwargs)
    elif model_name == 'dpn107':
        model = dpn107(
            num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool, **kwargs)
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
            num_classes=num_classes, pretrained=pretrained, transform_input=False, **kwargs)
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

    if resize_inputs:
        model = SelfResizingModel(model=model, input_size=input_size)

    if normalize_inputs:
        normalizer = get_normalizer(model_name_normalizer_name_mapping[model_name])
        model = NormalizedModel(model=model, normalizer=normalizer)

    if standardize_outputs:
        model = StandardizedOutputModel(model, drop_first_class=drop_first_class)

    return model

