import tensorflow as tf
import numpy as np
import functools

from deepmachine.flags import FLAGS
from .base import *
from ..models import stackedHG as models

slim = tf.contrib.slim


def DenseRegFace(
    inputs,
    is_training=True,
    deconv='bilinear',
    n_classes=FLAGS.quantization_step + 1,
    use_regression=False,
    **kwargs
):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[3]

    states = {}

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        with slim.arg_scope(models.hourglass_arg_scope_tf()):
            net = inputs
            # stacked hourglass

            _, net = models.hourglass(
                net,
                regression_channels=0,
                classification_channels=n_classes * 2,
                deconv=deconv)

            states['uv'] = net

            prediction = net

            return prediction, states


DenseFaceCascade = functools.partial(
    DenseIUVLandmark,
    n_landmarks=68
)
