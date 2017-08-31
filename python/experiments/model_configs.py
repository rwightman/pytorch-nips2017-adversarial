def config_from_string(string):
    string_to_config_dict = {
        'Inceptionv3': {'model_name': 'inception_v3', 'num_classes': 1000, 'input_size': 299,
                        'checkpoint_file': 'inception_v3_google-1a9a5a14.pth', 'drop_first_class': False},
        'Resnet18': {'model_name': 'resnet18-torchvision', 'num_classes': 1000, 'input_size': 224,
                     'checkpoint_file': 'resnet18-5c106cde.pth', 'drop_first_class': False},
        'Resnet34': {'model_name': 'resnet34-torchvision', 'num_classes': 1000, 'input_size': 224,
                     'checkpoint_file': 'resnet34-333f7ec4.pth', 'drop_first_class': False},
        'Resnet50': {'model_name': 'resnet50-torchvision', 'num_classes': 1000, 'input_size': 224,
                     'checkpoint_file': 'resnet50-19c8e357.pth', 'drop_first_class': False},
        'Resnet101': {'model_name': 'resnet101-torchvision', 'num_classes': 1000, 'input_size': 224,
                      'checkpoint_file': 'densenet121-241335ed.pth', 'drop_first_class': False},
        'Resnet152': {'model_name': 'resnet152-torchvision', 'num_classes': 1000, 'input_size': 224,
                      'checkpoint_file': 'resnet152-b121ed2d.pth', 'drop_first_class': False},
        'DenseNet121': {'model_name': 'densenet121-torchvision', 'num_classes': 1000, 'input_size': 224,
                        'checkpoint_file': 'densenet121-241335ed.pth', 'drop_first_class': False},
        'DenseNet169': {'model_name': 'densenet169-torchvision', 'num_classes': 1000, 'input_size': 224,
                        'checkpoint_file': 'densenet169-6f0f7f60.pth', 'drop_first_class': False},
        'DenseNet201': {'model_name': 'densenet201-torchvision', 'num_classes': 1000, 'input_size': 224,
                        'checkpoint_file': 'densenet201-4c113574.pth', 'drop_first_class': False},
        'DenseNet161': {'model_name': 'densenet161-torchvision', 'num_classes': 1000, 'input_size': 224,
                        'checkpoint_file': 'densenet161-17b70270.pth', 'drop_first_class': False},
        'SqueezeNet1_0': {'model_name': 'squeezenet1_0', 'num_classes': 1000, 'input_size': 224,
                          'checkpoint_file': 'squeezenet1_0-a815701f.pth', 'drop_first_class': False},
        'SqueezeNet1_1': {'model_name': 'squeezenet1_1', 'num_classes': 1000, 'input_size': 224,
                          'checkpoint_file': 'squeezenet1_1-f364aa15.pth', 'drop_first_class': False},
        'AlexNet': {'model_name': 'alexnet', 'num_classes': 1000, 'input_size': 224,
                    'checkpoint_file': 'alexnet-owt-4df8aa71.pth', 'drop_first_class': False},
        'DPN107Extra': {'model_name': 'dpn107', 'num_classes': 1000, 'input_size': 299,
                        'checkpoint_file': 'dpn107_extra-fc014e8ec.pth', 'drop_first_class': False},
        'DPN92Extra': {'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299,
                       'checkpoint_file': 'dpn92_extra-1f58102b.pth', 'drop_first_class': False},
        'DPN92': {'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299,
                  'checkpoint_file': 'dpn92-7d0f7156.pth', 'drop_first_class': False},
        'DPN68': {'model_name': 'dpn68', 'num_classes': 1000, 'input_size': 299,
                  'checkpoint_file': 'dpn68-abcc47ae.pth', 'drop_first_class': False},
        'AdvInceptionResnetV2': {'model_name': 'inception_resnet_v2', 'num_classes': 1001, 'input_size': 299,
                                 'checkpoint_file': 'adv_inception_resnet_v2.pth', 'drop_first_class': True},
        'InceptionResnetV2': {'model_name': 'inception_resnet_v2', 'num_classes': 1001, 'input_size': 299,
                              'checkpoint_file': 'inceptionresnetv2-d579a627.pth', 'drop_first_class': True},
    }

    for string, config_dict in string_to_config_dict.items():
        config_dict['pretrained'] = False
        config_dict['normalize_inputs'] = True
        config_dict['resize_inputs'] = True
        config_dict['standardize_outputs'] = True

    return string_to_config_dict[string]