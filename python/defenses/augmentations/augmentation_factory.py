from .blur import Blur, RandomBlur
from .crop import CentreCrop, RandomCrop
from .mirror import Mirror, RandomMirror
from .resize import Resize
from .normalize import get_normalizer
from collections import OrderedDict
import torch.nn as nn


#FIXME clean up and make more consistent interfaces for param/config, dicts?


def build_random_augmentation(
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


def build_fixed_augmentation(
        target_size,
        mirror=False,
        crop_ratio=1.0,
        blur=0,
        norm='torchvision'):
    seq = OrderedDict()
    if mirror:
        seq['mirror'] = Mirror()
    if crop_ratio != 1.0:
        seq['crop'] = CentreCrop(crop_ratio=crop_ratio)
    if target_size:
        seq['resize'] = Resize(target_size)
    if blur:
        seq['blur'] = Blur(k=blur)
    if norm:
        seq['norm'] = get_normalizer(norm)
    return nn.Sequential(seq)


def build_2crop_augmentation(
        target_size,
        crop_ratio=1.0,
        blur=0,
        norm='torchvision'):
    augs = []
    augs.append(build_fixed_augmentation(
        target_size, mirror=False, crop_ratio=crop_ratio, blur=blur, norm=norm))
    augs.append(build_fixed_augmentation(
        target_size, mirror=True, crop_ratio=crop_ratio, blur=blur, norm=norm))
    return augs


def build_4crop_augmentation(target_size, norm='torchvision'):
    augs = []
    augs.append(build_fixed_augmentation(
        target_size, mirror=False, crop_ratio=1.0, blur=2, norm=norm))
    augs.append(build_fixed_augmentation(
        target_size, mirror=True, crop_ratio=1.0, blur=2, norm=norm))
    augs.append(build_fixed_augmentation(
        target_size, mirror=False, crop_ratio=0.875,  blur=0, norm=norm))
    augs.append(build_fixed_augmentation(
        target_size, mirror=True, crop_ratio=0.9125, blur=0, norm=norm))
    return augs
