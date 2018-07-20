import tensorflow as tf
import numpy as np
from .. import layers


def ResiduleModule(inputs, out_channel, add_residule=False, kernel_initializer='glorot_uniform'):
    net = inputs
    for ch, kernal in zip([out_channel // 2, out_channel // 2, out_channel], [1, 3, 1]):
        net = layers.Conv2D(
            ch, (kernal, kernal),
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer)(net)
        net = layers.LeakyReLU()(net)

    if add_residule:
        res_net = layers.Conv2D(
            out_channel, (1, 1),
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer)(inputs)
        res_net = layers.LeakyReLU()(res_net)
        net = layers.Add()([net, res_net])

    return net


def Encoding2D(inputs, out_channel, down_sample=True, module='Residule', kernel_initializer='glorot_uniform', **kwargs):
    module_fn = globals()['%sModule' % module] if type(
        module) is str else module

    net = module_fn(inputs, out_channel,
                    kernel_initializer=kernel_initializer, **kwargs)

    if down_sample:
        net = layers.MaxPool2D(pool_size=2)(net)

    return net


def Decoding2D(inputs, out_channel, add_residule=False, kernel_initializer='glorot_uniform'):
    net = layers.Conv2DTranspose(
        out_channel, (3, 3), strides=2,
        padding='same',
        kernel_initializer=kernel_initializer)(inputs)
    net = layers.LeakyReLU()(net)

    return net


def Hourglass(inputs, output_shape, depth=4, initial_channel=32, module='Residule', kernel_initializer='glorot_uniform'):

    net = inputs
    net = layers.Conv2D(
        initial_channel, (7, 7),
        strides=2,
        padding='same',
        kernel_initializer=kernel_initializer)(net)
    net = layers.LeakyReLU()(net)
    net = layers.MaxPool2D(pool_size=2)(net)
    # Down sampling
    skip_layers = []
    for s in range(depth):
        skip_layers.append(net)
        net = Encoding2D(net, initial_channel * (s+1), down_sample=True,
                         add_residule=True, kernel_initializer=kernel_initializer)

    # feature compression
    net = Encoding2D(net, initial_channel * (depth + 1),
                     down_sample=False, kernel_initializer=kernel_initializer)

    # up sampling
    for s, s_layer in zip(range(depth), skip_layers[::-1]):
        net = Decoding2D(net, initial_channel * (depth-s),
                         kernel_initializer=kernel_initializer)
        net = layers.Concatenate()([net, s_layer])

    # output regress
    net = layers.Conv2DTranspose(
        initial_channel * (depth-s), (7, 7), strides=4,
        padding='same',
        kernel_initializer=kernel_initializer)(net)
    net = layers.LeakyReLU()(net)

    prediction = layers.Conv2D(
        output_shape[-1], (3, 3),
        padding='same',
        kernel_initializer=kernel_initializer)(net)

    return prediction


def Encoder2D(inputs, embedding, depth=2, conv_channel=32, kernel_initializer='glorot_uniform'):

    net = layers.Conv2D(
        conv_channel, (3, 3),
        padding='same',
        kernel_initializer=kernel_initializer)(inputs)
    net = layers.LeakyReLU()(net)

    for s in range(depth):

        net = Encoding2D(net, conv_channel * (s+2))

    net = Encoding2D(net, conv_channel * (depth+1), down_sample=False)

    net = layers.Flatten()(net)
    net = layers.Dense(embedding)(net)
    net = layers.LeakyReLU()(net)
    net = layers.Dropout(0.3)(net)

    return net


def Decoder2D(inputs, out_shape, depth=2, conv_channel=32, kernel_initializer='glorot_uniform'):

    input_shape = np.array(list(np.array(
        out_shape[:-1]) / np.power([2, 2], depth)) + [inputs.shape.as_list()[-1]]).astype(np.int)
    
    net = layers.Dense(np.prod(input_shape))(inputs)
    net = layers.LeakyReLU()(net)

    net = layers.Reshape(input_shape)(net)

    for s in range(depth):
        net = layers.Conv2D(
            conv_channel * (depth - s), (3, 3),
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer)(net)
        net = layers.LeakyReLU()(net)
        net = Decoding2D(net, conv_channel * (depth - s))

    net = layers.Conv2D(
        conv_channel, (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer)(net)
    net = layers.LeakyReLU()(net)
    net = layers.Conv2D(
        out_shape[-1], (3, 3),
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer)(net)
    net = layers.LeakyReLU()(net)

    return net


def AutoEncoder(inputs, output_shape, depth=2, embedding=128, initial_channel=32, kernel_initializer='glorot_uniform'):

    embedding = Encoder2D(inputs, embedding,
                          depth=depth, conv_channel=initial_channel)
    reconstruction = Decoder2D(
        embedding, output_shape, depth=depth, conv_channel=initial_channel)

    return reconstruction