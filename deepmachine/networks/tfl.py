import tensorflow as tf
import numpy as np
from functools import partial
from .. import layers

def ResiduleModule(x, out_channels, ks=3, s=1, activation=tf.nn.relu, **kwargs):
    in_channels = x.get_shape().as_list()[-1]

    # conv
    y = tf.layers.BatchNormalization()(x)
    y = tf.layers.conv2d(y, out_channels, ks, strides=1,
                         padding='same', activation=activation)
    y = tf.layers.BatchNormalization()(y)
    y = tf.layers.conv2d(y, out_channels, ks, strides=s,
                         padding='same', activation=activation)
    y = tf.layers.BatchNormalization()(y)

    # residule
    if in_channels != out_channels or s > 1:
        x = tf.layers.conv2d(x, out_channels, 1, strides=s,
                             padding='same', activation=None)

    return y + x


def Encoder2D(inputs, embedding, depth=4, nf=32, name='image_encoder', reuse=False, **kwargs):

    with tf.variable_scope(name, reuse=reuse):
        net = tf.layers.conv2d(
            inputs, nf, 3,
            activation=tf.nn.relu,
            padding='same', **kwargs)

        for s in range(1, depth):
            s = np.min([s, 4])
            net = ResiduleModule(net, nf * 2 ** s, s=2)

        s = np.min([depth, 4])
        net = ResiduleModule(net, nf * 2 ** s, s=1)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, rate=0.3)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, embedding)

    return net


def MeshEncoder(inputs, embeding, graph_laplacians, downsampling_matrices, polynomial_order=6, filter_list=[16, 16, 16, 32], name='mesh_encoder', reuse=False, **kwargs):
    with tf.variable_scope(name, reuse=reuse):
        net = inputs
        for nf, nl, nd in zip(filter_list, graph_laplacians, downsampling_matrices):

            net = layers.tfl.MeshConv(
                nl, nf=nf, polynomial_order=polynomial_order, **kwargs)(net)
            net = tf.layers.batch_normalization(net)
            net = layers.tfl.MeshReLU1B()(net)
            net = layers.tfl.MeshPool(nd)(net)

        # Fully connected hidden layers.
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, embeding)

    return net


def MeshDecoder(inputs, out_channel, graph_laplacians, adj_matrices, upsamling_matrices, polynomial_order=6, filter_list=[16, 16, 16, 16], name='mesh_decoder', reuse=False, **kwargs):
    with tf.variable_scope(name, reuse=reuse):
        pool_size = list(map(lambda x: x.shape[0], adj_matrices))
        net = inputs
        net = tf.layers.Dense(pool_size[-1] * filter_list[-1])(net)
        net = tf.reshape(net, [-1, pool_size[-1], filter_list[-1]])

        for nf, nl, nu in zip(filter_list[::-1], graph_laplacians[-2::-1], upsamling_matrices[::-1]):
            net = layers.tfl.MeshPoolTrans(nu)(net)
            net = layers.tfl.MeshConv(
                nl, nf=nf, polynomial_order=polynomial_order, **kwargs)(net)
            net = tf.layers.batch_normalization(net)
            net = layers.tfl.MeshReLU1B()(net)

        net = layers.MeshConv(graph_laplacians[0], nf=out_channel,
                       polynomial_order=polynomial_order, **kwargs)(net)

    return net