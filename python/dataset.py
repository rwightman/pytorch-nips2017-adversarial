import torch.utils.data as data
from torchvision import transforms

from PIL import Image
import os
import random
import torch
import pandas as pd

IMG_EXTENSIONS = ['.png', '.jpg']


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def default_inception_transform(img_size, scale=0.875):
    tf = transforms.Compose([
        transforms.Scale(round(img_size/scale)),
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


def find_inputs(folder, filename_to_target=None, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                target = filename_to_target[rel_filename] if filename_to_target else 0
                inputs.append((abs_filename, target))
    return inputs


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            target_file='target_class.csv',
            random_target=False,
            transform=transforms.ToTensor()):

        if target_file:
            target_df = pd.read_csv(os.path.join(root, target_file), header=None)
            f_to_t = dict(zip(target_df[0], target_df[1] - 1))  # -1 for 0-999 class ids
        else:
            f_to_t = dict()
        imgs = find_inputs(root, filename_to_target=f_to_t)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        if random_target:
            imgs = [(i, generate_target(l)) for i, l in imgs]

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

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]
