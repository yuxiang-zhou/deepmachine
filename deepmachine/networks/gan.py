import tensorflow as tf
import numpy as np

from ..flags import FLAGS
from ..models import gan
from ..models import stackedHG as shg

from .. import utils

slim = tf.contrib.slim


def GAN(
    inputs,
    is_training=True,
    n_channels=3,
    key_discriminant='inputs',
    **kwargs
):

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        states = {}

        states['generator'] = gan.generator_resnet(
            inputs, n_channels, reuse=False, name="generator")

        states['discriminator_pred'] = gan.discriminator(
            prediction, name="discriminator")

        if 'data_eps' in kwargs:
            inputs = kwargs['data_eps'][key_discriminant]

        states['discriminator_gt'] = gan.discriminator(
            inputs, reuse=True, name="discriminator")

        return prediction, states


def PoseGAN(
    inputs,
    is_training=True,
    n_channels=16,
    deconv='bilinear',
    bottleneck='bottleneck',
    **kwargs
):

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        states = {}

        with slim.arg_scope(gan.gan_arg_scope_tf()):
            pred_hm = prediction = states['generator'] = gan.generator(
                inputs, n_channels, deconv=deconv, bottleneck=bottleneck, reuse=False, name="generator")

        gt_hm = kwargs['data_eps']['heatmap']

        pred_lms = utils.tf_heatmap_to_lms(pred_hm)
        gt_lms = utils.tf_heatmap_to_lms(gt_hm)

        batch_size = tf.shape(inputs)[0]

        pred_patch = tf.map_fn(
            lambda x: utils.tf_image_patch_around_lms(inputs[x], pred_lms[x]),
            tf.range(batch_size),
            dtype=tf.float32
        )

        gt_patch = tf.map_fn(
            lambda x: utils.tf_image_patch_around_lms(inputs[x], gt_lms[x]),
            tf.range(batch_size),
            dtype=tf.float32
        )

        states['discriminator_pred'] = gan.discriminator(
            pred_patch, name="discriminator")

        states['discriminator_gt'] = gan.discriminator(
            gt_patch, reuse=True, name="discriminator")

        return prediction, states


def CycleGAN(inputs, is_training=True, n_channels=3, **kwargs):
    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
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


def CycleGANHG(inputs, is_training=True, n_channels=3, **kwargs):

    with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
        with slim.arg_scope(gan.gan_arg_scope_tf()):
            states = {}

            # inputs
            input_A = inputs[..., :n_channels]
            input_B = inputs[..., n_channels:]

            # generators
            states['fake_B'] = fake_B = gan.create_generator(
                input_A, n_channels, reuse=False, name="generatorAB")
            states['fake_A'] = fake_A = gan.create_generator(
                input_B, n_channels, reuse=False, name="generatorBA")
            states['rec_A'] = gan.create_generator(
                fake_B, n_channels, reuse=True, name="generatorBA")
            states['rec_B'] = gan.create_generator(
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
