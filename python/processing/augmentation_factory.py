from .blur import Blur, RandomBlur, GaussianBlur, RandomGaussianBlur
from .colour import RandomBrightnessContrast, RandomSaturation
from .crop import CentreCrop, RandomCrop
from .mirror import Mirror, RandomMirror
from .resize import Resize
from .normalize import get_normalizer
from collections import OrderedDict
import torch.nn as nn


#FIXME clean up and make more consistent interfaces for param/config, dicts?


def build_random_augmentation_module(
        target_size,
        mirror=RandomMirror(mirror_prob=0.5),
        crop=RandomCrop(crop_ratio=[0.875, 1.0]),
        blur=RandomBlur(blur_prob=0.5, blur2x2_prob=0.5),
        norm='torchvision'):

    seq = OrderedDict()
    if mirror is not None:
        seq['mirror'] = mirror
    if crop is not None:
        seq['crop'] = crop
    if target_size:
        seq['resize'] = Resize(target_size)
    if blur is not None:
        seq['blur'] = blur
    if norm:
        seq['norm'] = get_normalizer(norm)
    return nn.Sequential(seq)


def build_fixed_augmentation_module(
        target_size,
        mirror=False,
        crop_ratio=1.0,
        blur_type='',
        blur_k=None,
        blur_sigma=None,
        norm='torchvision'):
    seq = OrderedDict()
    if mirror:
        seq['mirror'] = Mirror()
    if crop_ratio != 1.0:
        seq['crop'] = CentreCrop(crop_ratio=crop_ratio)
    if target_size:
        seq['resize'] = Resize(target_size)
    if blur_type == 'g' or blur_type == 'gaussian':
        seq['blur'] = GaussianBlur(kernel_size=blur_k, sigma=blur_sigma)
    elif blur_type == 'm' or blur_type == 'median':
        seq['blur'] = Blur(k=blur_k)
    if norm:
        seq['norm'] = get_normalizer(norm)
    return nn.Sequential(seq)


def build_anp_augmentation_module(
        mirror_prob=0.5,
        gaus_blur_prob=0.5,
        gaus_blur_size=3,
        gaus_blur_sigma=.5,
        brightness_contrast=False,
        saturation=False):
    modules = [RandomMirror(mirror_prob)]
    if brightness_contrast:
        modules.append(RandomBrightnessContrast())
    if saturation:
        modules.append(RandomSaturation())
    modules.extend([
        RandomGaussianBlur(gaus_blur_prob, gaus_blur_size, gaus_blur_sigma),
        RandomCrop()
    ])
    return nn.Sequential(*modules)


def build_2crop_augmentation(
        target_size,
        crop_ratio=1.0,
        blur_type='m',
        blur_k=0,
        norm='torchvision'):
    augs = []
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=False, crop_ratio=crop_ratio, blur_type=blur_type, blur_k=blur_k, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=True, crop_ratio=crop_ratio, blur_type=blur_type, blur_k=blur_k, norm=norm))
    return augs


def build_4crop_augmentation(target_size, norm='torchvision'):
    augs = []
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=False, crop_ratio=1.0, blur_type='m', blur_k=2, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=True, crop_ratio=1.0, blur_type='m', blur_k=2, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=False, crop_ratio=0.875, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=True, crop_ratio=0.9125, norm=norm))
    return augs


def build_8crop_augmentation(target_size, norm='torchvision'):
    augs = []
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=False, crop_ratio=1.0, blur_type='m', blur_k=3, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=True, crop_ratio=1.0,  blur_type='g', blur_k=3, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=False, crop_ratio=0.875,  blur_type='g', blur_k=3, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=True, crop_ratio=0.875,  blur_type='m', blur_k=2, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=False, crop_ratio=0.9167,  blur_type='m', blur_k=2, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=True, crop_ratio=0.9167,  blur_type='g', blur_k=3, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=False, crop_ratio=0.9583,  blur_type='g', blur_k=3, norm=norm))
    augs.append(build_fixed_augmentation_module(
        target_size, mirror=True, crop_ratio=0.9583,  blur_type='m', blur_k=3, norm=norm))
    return augs



