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
from .features import *

slim = tf.contrib.slim


DenseRegPoseProvider = functools.partial(
    TFRecordNoFlipProvider,
    features=FeatureIUV,
    augmentation=True,
    resolvers=ResolverIUV
)

DensePoseProvider = functools.partial(
    TFRecordNoFlipProvider,
    features=FeatureIUVHM,
    augmentation=True,
    resolvers=ResolverIUVHM
)

HeatmapProvider = functools.partial(
    TFRecordProvider,
    features=FeatureHeatmap,
    augmentation=True,
    resolvers=ResolverHMPose
)

BBoxHeatmapProvider = functools.partial(
    TFRecordBBoxProvider,
    features=FeatureRLMS,
    augmentation=True,
    resolvers=ResolverBBoxPose
)
