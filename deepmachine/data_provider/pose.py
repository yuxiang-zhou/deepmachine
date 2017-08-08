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
from .base import TFRecordProvider, image_resolver, heatmap_resolver, iuv_resolver
slim = tf.contrib.slim


FeatureIUV = {
    # images
    'image': tf.FixedLenFeature([], tf.string),
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    # iuv
    'iuv': tf.FixedLenFeature([], tf.string),
    'iuv_height': tf.FixedLenFeature([], tf.int64),
    'iuv_width': tf.FixedLenFeature([], tf.int64),
    # svs
    'n_svs': tf.FixedLenFeature([], tf.int64),
    'n_svs_ch': tf.FixedLenFeature([], tf.int64),
    'svs': tf.FixedLenFeature([], tf.string),
    # landmarks
    'n_landmarks': tf.FixedLenFeature([], tf.int64),
    'gt': tf.FixedLenFeature([], tf.string),
    'visible': tf.FixedLenFeature([], tf.string),
    'marked': tf.FixedLenFeature([], tf.string),
    'scale': tf.FixedLenFeature([], tf.float32),
    # original infomations
    'original_scale': tf.FixedLenFeature([], tf.float32),
    'original_centre': tf.FixedLenFeature([], tf.string),
    'original_lms': tf.FixedLenFeature([], tf.string),
    # inverse transform to original landmarks
    'restore_translation': tf.FixedLenFeature([], tf.string),
    'restore_scale': tf.FixedLenFeature([], tf.float32)
}

FeatureHeatmap = {
    # images
    'image': tf.FixedLenFeature([], tf.string),
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),

    # landmarks
    'n_landmarks': tf.FixedLenFeature([], tf.int64),
    'gt': tf.FixedLenFeature([], tf.string),
    'visible': tf.FixedLenFeature([], tf.string),
    'marked': tf.FixedLenFeature([], tf.string),
}

ResolverHM = {
    'inputs': image_resolver,
    'heatmap': heatmap_resolver,
}

ResolverIUV = {
    'inputs': image_resolver,
    'heatmap': heatmap_resolver,
    'iuv': iuv_resolver
}


class TFRecordIUVProvider(TFRecordProvider):

    # flip, rotate, scale
    def _random_augmentation(self):
        return tf.concat([tf.random_uniform([1]) - 1,
                          (tf.random_uniform([1]) * 60. - 30.) * np.pi / 180.,
                          tf.random_uniform([1]) * 0.5 + 0.75], 0)


DensePoseProvider = functools.partial(
    TFRecordIUVProvider,
    features=FeatureIUV,
    augmentation=True,
    resolvers=ResolverIUV
)

HeatmapProvider = functools.partial(
    TFRecordProvider,
    features=FeatureHeatmap,
    augmentation=True,
    resolvers=ResolverHM
)
