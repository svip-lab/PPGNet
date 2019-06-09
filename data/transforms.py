from torchvision.transforms import functional as tf
import numpy as np
from PIL import Image
import random
from functools import partial


class Compose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, img, pt):
        for t in self.transforms:
            img, pt = t(img, pt)

        return img, pt


class RandomCompose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, img, pt):
        random.shuffle(self.transforms)
        for t in self.transforms:
            img, pt = t(img, pt)

        return img, pt        


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img, pt):
        w, h = img.size
        y_scale = (self.size[0] - 1) / (h - 1)
        x_scale = (self.size[1] - 1) / (w - 1)

        pt_new = np.zeros_like(pt)
        pt_mask = pt.sum(axis=1) > 0
        pt_new[pt_mask] = np.vstack((pt[pt_mask][:, 0] * x_scale, pt[pt_mask][:, 1] * y_scale)).T

        assert not pt_new.sum() == 0

        return img.resize(self.size[::-1], self.interpolation), pt_new


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pt):
        if self.p > np.random.rand():
            w, _ = img.size
            img = tf.hflip(img)
            pt_new = np.zeros_like(pt)
            pt_mask = pt.sum(axis=1) > 0
            pt_new[pt_mask] = np.vstack((w - 1 - pt[pt_mask][:, 0], pt[pt_mask][:, 1])).T
            return img, pt_new
        return img, pt


class RandomColorAug(object):
    def __init__(self, factor=0.2):
        self.factor = factor

    def __call__(self, img, pt):
        transforms = [
            tf.adjust_brightness,
            tf.adjust_contrast,
            tf.adjust_saturation
            ]
        random.shuffle(transforms)
        for t in transforms:
            img = t(img, (np.random.rand() - 0.5) * 2 * self.factor + 1)

        return img, pt