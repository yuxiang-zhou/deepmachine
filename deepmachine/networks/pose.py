import tensorflow as tf
import numpy as np
import functools

from ..flags import FLAGS
from ..models import stackedHG as models

from .base import *

slim = tf.contrib.slim


def DensePoseMix(
    inputs,
    is_training=True,
    deconv='bilinear',
    n_features=26 * 3,
    **kwargs
):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[3]

    states = {}

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        with slim.arg_scope(models.hourglass_arg_scope_tf()):
            # first hourglass
            net = inputs
            net = models.StackedHourglassTorch(net, n_features, deconv=deconv)

            states['uv'] = net

            # second hourglass

            net = tf.concat((inputs, net), 3)
            prediction = models.StackedHourglassTorch(net, 16, deconv=deconv)

            return prediction, states


def DensePoseTorch(
    inputs,
    is_training=True,
    deconv='bilinear',
    n_features=26 * 3,
    **kwargs
):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[3]

    states = {}

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        with slim.arg_scope(models.hourglass_arg_scope_torch()):
            # first hourglass
            net = inputs
            net = models.StackedHourglassTorch(net, n_features, deconv=deconv)

            states['uv'] = net

            # second hourglass

            net = tf.concat((inputs, net), 3)
            prediction = models.StackedHourglassTorch(net, 16, deconv=deconv)

            return prediction, states


DensePoseTF = functools.partial(
    DenseIUVLandmark,
    n_landmarks=16
)

def DenseRegPose(
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
                classification_channels=n_classes*3,
                deconv=deconv)

            states['uv'] = net

            prediction = net

            return prediction, states