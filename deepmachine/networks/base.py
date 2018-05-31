import tensorflow as tf
import numpy as np

from deepmachine.flags import FLAGS
import deepmachine.models.stackedHG as models
from deepmachine.models import autoencoder

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

def StackedHourglassAE(
    inputs,
    is_training=True,
    deconv='transpose+conv+relu',
    n_channels=16,
    n_stacks=2,
    bottleneck='bottleneck',
    **kwargs
):
    batch_size, height, width, channels = inputs.get_shape().as_list()

    with tf.variable_scope('regression'):
        net, states = StackedHourglass(
            inputs,
            is_training=True,
            deconv=deconv,
            n_channels=n_channels,
            n_stacks=n_stacks,
            bottleneck=bottleneck,
            **kwargs)

        embedding_input = autoencoder.encoder(net, out_channel=512, scope='encoder_input')
        embedding_input = tf.identity(embedding_input, 'embedding_input')

    with tf.variable_scope('reconstruction'):
        gt_hm = kwargs['data_eps']['heatmap']
        embedding_ae = autoencoder.encoder(gt_hm, out_channel=512, scope='encoder_ae')
        embedding_ae = tf.identity(embedding_ae, 'embedding_ae')

        reconstruction = autoencoder.decoder(embedding_ae, out_channel=n_channels, scope='decoder', reuse=False)
        prediction = autoencoder.decoder(embedding_input, out_channel=n_channels, scope='decoder', reuse=True)

    states += [embedding_input, embedding_ae, prediction, reconstruction]

    return prediction, states


def AutoEncoder(
    inputs,
    is_training=True,
    deconv='transpose+conv+relu',
    n_channels=16,
    **kwargs
):

    inputs = kwargs['data_eps']['heatmap']
    with tf.variable_scope('reconstruction'):
        net = autoencoder.encoder(inputs, out_channel=512, scope='encoder_input')
        net = tf.identity(net, 'embedding')

        states = [net]

        prediction = autoencoder.decoder(net, out_channel=n_channels, scope='decoder')

        states += [prediction]

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

            states['uv'] = tf.identity(net, name='uv')

            # second hourglass

            net = tf.concat((inputs, net), 3)
            with tf.variable_scope('lms_regression'):
                net, _ = models.hourglass(
                    net,
                    regression_channels=n_landmarks,
                    classification_channels=0,
                    deconv=deconv)

            states['heatmap'] = tf.identity(net, name='heatmap')
            prediction = net

            return prediction, states
