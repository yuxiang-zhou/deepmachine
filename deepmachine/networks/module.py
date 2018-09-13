import tensorflow as tf
import numpy as np
import keras.backend as K
from functools import partial
from keras.utils import get_custom_objects
from .. import layers

def _conv(inputs, conv_layer, *args, padding='same', batch_norm=None, dropout=None, pre_conv=None, activation=True, **kwargs):

    net = inputs

    # pad coordinate
    if pre_conv:
        net = pre_conv()(net)

    # convolution layer
    net = conv_layer(*args, padding=padding, **kwargs)(net)

    # rectifier
    if activation:
        if type(activation) is str:
            net = layers.Activation(activation)(net)
        else:
            net = layers.LeakyReLU()(net)

    # batch normalization
    if batch_norm:
        if type(batch_norm) is str:
            net = getattr(layers, batch_norm)()(net)
        else:
            net = layers.BatchNormalization()(net)

    # dropout layer
    if dropout:
        net = layers.Dropout(dropout)(net)

    return net


def conv2d(inputs, *args, use_coordconv=False, **kwargs):

    if use_coordconv:
        kwargs['pre_conv'] = layers.CoordinateChannel2D

    return _conv(inputs, layers.Conv2D, *args, **kwargs)


def deconv2d(inputs, *args, use_coordconv=False, size=2, **kwargs):

    kwargs['pre_conv'] = partial(layers.UpSampling2D, size=size)

    return _conv(inputs, layers.Conv2D, *args, **kwargs)


def conv2dt(inputs, nf, *args, use_coordconv=False, **kwargs):

    if use_coordconv:
        kwargs['pre_conv'] = layers.CoordinateChannel2D

    net = _conv(inputs, layers.Deconv2D, nf, *args, **kwargs)

    return net



def vae_sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


get_custom_objects().update({'vae_sampling': vae_sampling})