import os
import re
import torch
import pandas as pd
import random

import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def default_inception_transform(img_size):
    tf = transforms.Compose([
        transforms.Scale(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    return tf


def generate_target(ol, num_classes=1000):
    rl = random.randrange(0, num_classes)
    while rl == ol:
        rl = random.randint(0, num_classes)
    return rl


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    if class_to_idx is None:
        class_to_idx = dict()
        build_class_idx = True
    else:
        build_class_idx = False
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        if build_class_idx and not subdirs:
            class_to_idx[label] = None
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if build_class_idx:
        classes = sorted(class_to_idx.keys(), key=natural_key)
        for idx, c in enumerate(classes):
            class_to_idx[c] = idx
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    if build_class_idx:
        return images_and_targets, classes, class_to_idx
    else:
        return images_and_targets


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            target_file='target_class.csv',
            transform=None,
            random_target=False):

        if target_file:
            target_df = pd.read_csv(os.path.join(root, target_file), header=None)
            f_to_t = dict(zip(target_df[0], target_df[1] - 1))  # -1 for 0-999 class ids
            imgs = find_images_and_targets(root, class_to_idx=f_to_t)
        else:
            imgs, _, _ = find_images_and_targets(root, class_to_idx=None)

        if random_target:
            imgs = [(i, generate_target(l)) for i, l in imgs]

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def set_transform(self, transform):
        self.transform = transform

    def filenames(self, indices=[], basename=False, relname=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            elif relname:
                return [os.path.relpath(self.imgs[i][0], self.root) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            elif relname:
                [os.path.relpath(x[0], self.root) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]
