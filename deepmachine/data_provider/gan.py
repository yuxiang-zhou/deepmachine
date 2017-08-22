import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
import scipy
import functools

from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation

import sys

from ..flags import FLAGS
from ..utils import tf_lms_to_heatmap, tf_rotate_points
from .base import *

slim = tf.contrib.slim


def dummy_resolver(_, *args, **kwargs):
    dummy = tf.constant(np.random.sample([1]).astype(np.float32))
    dummy.set_shape([1])

    return dummy


def cyclegan_image_file_resolver(content, aug=False, aug_args=tf.constant([0, 0, 1])):
    image = tf.image.decode_png(content)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_channels = tf.shape(image)[2]

    image = tf.cond(image_channels > 1,
                    lambda: image,
                    lambda: tf.image.grayscale_to_rgb(image))
    image = tf.to_float(image) / 255. * 2 - 1

    # augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_images(image, [286, 286])
    image = tf.random_crop(image, [256, 256, 3])

    # shape defination
    image.set_shape([None, None, 3])

    return image


ResolverImage = {
    'inputs': cyclegan_image_file_resolver,
    'dummy': dummy_resolver
}


def CycleGanProvider(path, *args, **kwargs):
    _paths = path.split(':')

    _p1 = TFDirectoryProvider(
        dirpath=_paths[0],
        augmentation=True,
        image_size=256,
        ext='.jpg',
        no_processes=4,
        resolvers=ResolverImage)

    _p2 = TFDirectoryProvider(
        dirpath=_paths[1],
        augmentation=True,
        image_size=256,
        ext='.jpg',
        no_processes=4,
        resolvers=ResolverImage)

    return DatasetPairer([_p1, _p2])
