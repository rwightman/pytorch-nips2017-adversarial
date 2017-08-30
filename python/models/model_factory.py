from torchvision.models import *
from .resnext101_32x4d import resnext101_32x4d
from .inception_v4 import inception_v4
from .inception_resnet_v2 import inception_resnet_v2
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .my_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .fbresnet200 import fbresnet200
from .dpn import dpn68, dpn92, dpn98, dpn131, dpn107


def create_model(model_name='resnet50', pretrained=True, num_classes=1000, **kwargs):
    if 'test_time_pool' in kwargs:
        test_time_pool = kwargs.pop('test_time_pool')
    else:
        test_time_pool = True

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
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet161':
        model = densenet161(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet169':
        model = densenet169(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet201':
        model = densenet201(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_v3':
        model = inception_v3(
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
    else:
        assert False and "Invalid model"

    return model

