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
from .resolvers import *

slim = tf.contrib.slim



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
