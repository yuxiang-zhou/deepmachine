import tensorflow as tf
import numpy as np

from deepmachine.flags import FLAGS
import deepmachine.models.stackedHG as models

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

    states = []

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        with slim.arg_scope(models.hourglass_arg_scope_tf()):
            net = inputs
            # stacked hourglass

            _, net = models.hourglass(
                net,
                regression_channels=0,
                classification_channels=n_classes*2,
                deconv=deconv)

            states.append(net)

            net, _ = models.hourglass(
                net,
                regression_channels=68,
                classification_channels=0,
                deconv=deconv)


            prediction = net

            return prediction, states
