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


FeatureSequence = {
    # sequences
    'frames': tf.VarLenFeature(tf.string),
    'drawings': tf.VarLenFeature(tf.string),

    # meta data
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
}

FeatureMaskedSequence = {
    # sequences
    'frames': tf.VarLenFeature(tf.string),
    'drawings': tf.VarLenFeature(tf.string),
    'masks': tf.FixedLenFeature([], tf.string),

    # meta data
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
}

FeatureIUVHM = {
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

FeatureIUV = {
    # images
    'image': tf.FixedLenFeature([], tf.string),
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    # iuv
    'iuv': tf.FixedLenFeature([], tf.string),
    'iuv_height': tf.FixedLenFeature([], tf.int64),
    'iuv_width': tf.FixedLenFeature([], tf.int64),
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


FeatureRLMS = {
    # images
    'image': tf.FixedLenFeature([], tf.string),
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),

    # landmarks
    'n_landmarks': tf.FixedLenFeature([], tf.int64),
    'rlms': tf.FixedLenFeature([], tf.string),
    'visible': tf.FixedLenFeature([], tf.string),
    'marked': tf.FixedLenFeature([], tf.string),
}