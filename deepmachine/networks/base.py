import tensorflow as tf
import numpy as np

from deepmachine.flags import FLAGS
import deepmachine.models.stackedHG as models

slim = tf.contrib.slim


def StackedHourglass(
    inputs,
    is_training=True,
    deconv='bilinear',
    n_channels=16,
    n_stacks=2,
    bottleneck='bottleneck',
    **kwargs
):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[3]

    states = []

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        with slim.arg_scope(models.hourglass_arg_scope_tf()):
            net = None
            # stacked hourglass
            for i in range(n_stacks):
                with tf.variable_scope('stack_%02d' % i):
                    if net is not None:
                        net = tf.concat((inputs, net), 3)
                    else:
                        net = inputs

                    net, _ = models.hourglass(
                        net,
                        regression_channels=n_channels,
                        classification_channels=0,
                        deconv=deconv,
                        bottleneck=bottleneck)

                    states.append(net)

            prediction = net

            return prediction, states


def DenseIUVLandmark(
    inputs,
    is_training=True,
    deconv='bilinear',
    n_features=26 * 3,
    n_landmarks=16,
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
            with tf.variable_scope('iuv_regression'):
                net, _ = models.hourglass(
                    net,
                    regression_channels=n_features,
                    classification_channels=0,
                    deconv=deconv)

            states['uv'] = net

            # second hourglass

            net = tf.concat((inputs, net), 3)
            with tf.variable_scope('lms_regression'):
                net, _ = models.hourglass(
                    net,
                    regression_channels=n_landmarks,
                    classification_channels=0,
                    deconv=deconv)

            states['heatmap'] = net
            prediction = net

            return prediction, states
