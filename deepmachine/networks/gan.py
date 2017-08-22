import tensorflow as tf
import numpy as np

from ..flags import FLAGS
from ..models import gan
slim = tf.contrib.slim


def GAN(
    inputs,
    is_training=True,
    deconv='bilinear',
    n_channels=16,
    n_stacks=2,
    bottleneck='bottleneck',
    key_discriminant='inputs',
    **kwargs
):

    states = {}

    with tf.variable_scope('generator'):
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
            with slim.arg_scope(gan.gan_arg_scope_tf()):
                net = None

                states['generator'] = []
                # stacked hourglass
                for i in range(n_stacks):
                    with tf.variable_scope('stack_%02d' % i):
                        if net is not None:
                            net = tf.concat((inputs, net), 3)
                        else:
                            net = inputs

                        net, _ = gan.generator(
                            net,
                            n_channels,
                            deconv=deconv,
                            bottleneck=bottleneck)

                        states['generator'].append(net)
                prediction = net

    discriminator_pred = gan.discriminator(prediction)
    states['discriminator_pred'] = discriminator_pred

    if 'data_eps' in kwargs:
        inputs = kwargs['data_eps'][key_discriminant]

    discriminator_gt = gan.discriminator(inputs, reuse=True)
    states['discriminator_gt'] = discriminator_gt

    return prediction, states


def CycleGAN(inputs, is_training=True, n_channels=3, **kwargs):
    states = {}

    # inputs
    input_A = inputs[..., :n_channels]
    input_B = inputs[..., n_channels:]

    # generators
    states['fake_B'] = fake_B = gan.generator_resnet(
        input_A, n_channels, reuse=False, name="generatorAB")
    states['fake_A'] = fake_A = gan.generator_resnet(
        input_B, n_channels, reuse=False, name="generatorBA")
    states['rec_A'] = gan.generator_resnet(
        fake_B, n_channels, reuse=True, name="generatorBA")
    states['rec_B'] = gan.generator_resnet(
        fake_A, n_channels, reuse=True, name="generatorAB")

    # discriminators
    states['disc_A_real'] = gan.discriminator(
        input_A, reuse=False, name='discriminatorA')
    states['disc_A_fake'] = gan.discriminator(
        fake_A, reuse=True, name='discriminatorA')
    states['disc_B_real'] = gan.discriminator(
        input_B, reuse=False, name='discriminatorB')
    states['disc_B_fake'] = gan.discriminator(
        fake_B, reuse=True, name='discriminatorB')

    prediction = tf.concat([states['fake_B'], states['fake_A']], 3)

    return prediction, states
