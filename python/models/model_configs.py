
def normalizer_from_model(model_name):
    if 'inception' in model_name:
        normalizer = 'le'
    elif 'dpn' in model_name:
        normalizer = 'dpn'
    else:
        normalizer = 'torchvision'
    return normalizer


def config_from_string(string, output_fn='log_softmax'):
    string_to_config_dict = {
        'inception_v3_tf': {
            'model_name': 'inception_v3', 'num_classes': 1001, 'input_size': 299,
            'checkpoint_file': 'inception_v3_rw.pth',  'drop_first_class': True, 'kwargs': {'aux_logits': False}},
        'adv_inception_v3': {
            'model_name': 'inception_v3', 'num_classes': 1001, 'input_size': 299,
            'checkpoint_file': 'adv_inception_v3_rw.pth', 'drop_first_class': True, 'kwargs': {'aux_logits': False}},
        'inception_v3': {
            'model_name': 'inception_v3', 'num_classes': 1000, 'input_size': 299,
            'checkpoint_file': 'inception_v3_google-1a9a5a14.pth', 'drop_first_class': False},
        'resnet18': {
            'model_name': 'resnet18', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'resnet18-5c106cde.pth', 'drop_first_class': False},
        'resnet34': {
            'model_name': 'resnet34', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'resnet34-333f7ec4.pth', 'drop_first_class': False},
        'resnet50': {
            'model_name': 'resnet50', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'resnet50-19c8e357.pth', 'drop_first_class': False},
        'resnet101': {
            'model_name': 'resnet101', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'resnet101-5d3b4d8f.pth', 'drop_first_class': False},
        'resnet152': {
            'model_name': 'resnet152', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'resnet152-b121ed2d.pth', 'drop_first_class': False},
        'densenet121': {
            'model_name': 'densenet121', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'densenet121-241335ed.pth', 'drop_first_class': False},
        'densenet169': {
            'model_name': 'densenet169', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'densenet169-6f0f7f60.pth', 'drop_first_class': False},
        'densenet201': {
            'model_name': 'densenet201', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'densenet201-4c113574.pth', 'drop_first_class': False},
        'densenet161': {
            'model_name': 'densenet161', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'densenet161-17b70270.pth', 'drop_first_class': False},
        'squeezenet1_0': {
            'model_name': 'squeezenet1_0', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'squeezenet1_0-a815701f.pth', 'drop_first_class': False},
        'squeezenet1_1': {
            'model_name': 'squeezenet1_1', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'squeezenet1_1-f364aa15.pth', 'drop_first_class': False},
        'alexnet': {
            'model_name': 'alexnet', 'num_classes': 1000, 'input_size': 224,
            'checkpoint_file': 'alexnet-owt-4df8aa71.pth', 'drop_first_class': False},
        'dpn107': {
            'model_name': 'dpn107', 'num_classes': 1000, 'input_size': 299,
            'checkpoint_file': 'dpn107_extra-fc014e8ec.pth', 'drop_first_class': False},
        'dpn92_extra': {
            'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299,
            'checkpoint_file': 'dpn92_extra-1f58102b.pth', 'drop_first_class': False},
        'dpn92': {
            'model_name': 'dpn92', 'num_classes': 1000, 'input_size': 299,
            'checkpoint_file': 'dpn92-7d0f7156.pth', 'drop_first_class': False},
        'dpn68': {
            'model_name': 'dpn68', 'num_classes': 1000, 'input_size': 299,
            'checkpoint_file': 'dpn68-abcc47ae.pth', 'drop_first_class': False},
        'dpn68b_extra': {
            'model_name': 'dpn68b', 'num_classes': 1000, 'input_size': 299,
            'checkpoint_file': 'dpn68_extra.pth', 'drop_first_class': False},
        'adv_inception_resnet_v2': {
            'model_name': 'inception_resnet_v2', 'num_classes': 1001, 'input_size': 299,
            'checkpoint_file': 'adv_inception_resnet_v2.pth', 'drop_first_class': True},
        'inception_resnet_v2': {
            'model_name': 'inception_resnet_v2', 'num_classes': 1001, 'input_size': 299,
            'checkpoint_file': 'inceptionresnetv2-d579a627.pth', 'drop_first_class': True},
    }

    for _, config_dict in string_to_config_dict.items():
        config_dict['normalizer'] = normalizer_from_model(config_dict['model_name'])
        config_dict['output_fn'] = output_fn

    return string_to_config_dict[string]
