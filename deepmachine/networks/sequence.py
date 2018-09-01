import tensorflow as tf
import numpy as np
from functools import partial
from .. import layers


def _conv(inputs, conv_layer, *args, batch_norm=None, dropout=None, pre_conv=None, activation=True, **kwargs):

    net = inputs

    # pad coordinate
    if pre_conv:
        net = pre_conv()(net)

    # convolution layer
    net = conv_layer(*args, **kwargs)(net)

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


def conv2dt(inputs, *args, batch_norm=None, use_coordconv=False, dropout=False, **kwargs):

    if use_coordconv:
        kwargs['pre_conv'] = layers.CoordinateChannel2D

    return _conv(inputs, layers.Conv2DTranspose, *args, **kwargs)


def ResiduleModule(inputs, out_channel, add_residule=False, kernel_initializer='glorot_uniform', **kwargs):
    net = inputs
    for ch, kernal in zip([out_channel // 2, out_channel // 2, out_channel], [1, 3, 1]):
        net = conv2d(
            net,
            ch, (kernal, kernal),
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer, **kwargs)

    if add_residule:
        res_net = conv2d(
            inputs,
            out_channel, (1, 1),
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer, **kwargs)
        net = layers.Add()([net, res_net])

    return net


def Encoding2D(inputs, out_channel, down_sample=True, module='Residule', add_residule=True, kernel_initializer='glorot_uniform', **kwargs):
    module_fn = globals()['%sModule' % module] if type(
        module) is str else module

    net = module_fn(inputs, out_channel,
                    kernel_initializer=kernel_initializer,
                    add_residule=True,
                    **kwargs)

    if down_sample:
        net = conv2d(net, out_channel, 3, strides=2, padding='same',
                     kernel_initializer=kernel_initializer, **kwargs)

    return net


def Decoding2D(inputs, out_channel, kernel_initializer='glorot_uniform', **kwargs):
    net = deconv2d(
        inputs,
        out_channel, (3, 3),
        padding='same',
        kernel_initializer=kernel_initializer, **kwargs)

    return net


def Hourglass(inputs, output_shape, depth=4, initial_channel=32, module='Residule', kernel_initializer='glorot_uniform', **kwargs):

    net = inputs
    net = conv2d(
        net,
        initial_channel, (7, 7),
        strides=2,
        padding='same',
        batch_norm=None,
        activation=True,
        dropout=None,
        use_coordconv=kwargs['use_coordconv'] or False,
        kernel_initializer=kernel_initializer)
    net = layers.MaxPool2D(pool_size=2)(net)
    # Down sampling
    skip_layers = []
    for s in range(depth):
        skip_layers.append(net)
        net = Encoding2D(net, initial_channel * (s+1), down_sample=True,
                         add_residule=True, kernel_initializer=kernel_initializer, module=module, **kwargs)

    # feature compression
    net = Encoding2D(net, initial_channel * (depth + 1),
                     down_sample=False, kernel_initializer=kernel_initializer, module=module, **kwargs)

    # up sampling
    for s, s_layer in zip(range(depth), skip_layers[::-1]):
        net = Decoding2D(net, initial_channel * (depth-s),
                         kernel_initializer=kernel_initializer, **kwargs)
        net = layers.Concatenate()([net, s_layer])

    # output regress
    net = deconv2d(
        net,
        initial_channel * (depth-s),
        kernel_size=3,
        size=4,
        padding='same',
        kernel_initializer=kernel_initializer, **kwargs)

    prediction = conv2d(
        net,
        output_shape[-1],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=kernel_initializer)

    return prediction


def Encoder2D(inputs, embedding, depth=2, conv_channel=32, kernel_initializer='glorot_uniform', **kwargs):

    net = conv2d(
        inputs,
        conv_channel, (3, 3),
        padding='same',
        kernel_initializer=kernel_initializer, **kwargs)

    for s in range(depth):

        net = Encoding2D(net, conv_channel * (s+2), **kwargs)

    net = Encoding2D(net, conv_channel * (depth+1),
                     down_sample=False, **kwargs)

    net = layers.Flatten()(net)
    net = layers.Dense(embedding)(net)
    net = layers.LeakyReLU()(net)
    net = layers.Dropout(0.3)(net)

    return net


def Decoder2D(inputs, out_shape, depth=2, conv_channel=32, kernel_initializer='glorot_uniform', **kwargs):

    input_shape = np.array(list(np.array(
        out_shape[:-1]) / np.power([2, 2], depth)) + [inputs.shape.as_list()[-1]]).astype(np.int)

    net = layers.Dense(np.prod(input_shape))(inputs)
    net = layers.LeakyReLU()(net)

    net = layers.Reshape(input_shape)(net)

    for s in range(depth):
        net = conv2d(
            net,
            conv_channel * (depth - s), (3, 3),
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer, **kwargs)
        net = Decoding2D(net, conv_channel * (depth - s), **kwargs)

    net = conv2d(
        net,
        conv_channel, (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer, **kwargs)

    net = conv2d(
        net,
        out_shape[-1], (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer, **kwargs)

    return net


def AutoEncoder(inputs, output_shape, depth=2, embedding=128, initial_channel=32, kernel_initializer='glorot_uniform', **kwargs):

    embedding = Encoder2D(inputs, embedding,
                          depth=depth, conv_channel=initial_channel, **kwargs)
    reconstruction = Decoder2D(
        embedding, output_shape, depth=depth, conv_channel=initial_channel, **kwargs)

    return reconstruction
