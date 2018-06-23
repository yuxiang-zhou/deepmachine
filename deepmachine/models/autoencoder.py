import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from .stackedHG import hourglass_arg_scope_tf, bottleneck_module, deconv_layer, hourglass

def encoder(inputs, out_channel=64, conv_channel=64, reuse=False, scope='encoder', is_training=True):
    batch_size, input_h, input_w, _ = inputs.get_shape().as_list()
    
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        with slim.arg_scope(
            [slim.conv2d], 
            activation_fn=tf.nn.leaky_relu,
            normalizer_fn=None,
            normalizer_params={'is_training': is_training},
            stride=1
        ):

            net = slim.conv2d(inputs, conv_channel, [3, 3])

            net = slim.conv2d(net, conv_channel, [3, 3])
            net = slim.conv2d(net, conv_channel * 2, [3, 3])
            net = slim.max_pool2d(inputs, [2, 2])

            net = slim.conv2d(net, conv_channel * 2, [3, 3])
            net = slim.conv2d(net, conv_channel * 3, [3, 3])
            net = slim.max_pool2d(inputs, [2, 2])

            net = slim.conv2d(net, conv_channel * 3, [3, 3])
            net = slim.conv2d(net, conv_channel * 3, [3, 3])

            net = slim.flatten(net)
            net = slim.dropout(net)
            net = slim.fully_connected(net, out_channel, activation_fn=tf.nn.leaky_relu)
    
    return net

def decoder(inputs, out_channel=3, conv_channel=64, reuse=False, scope='decoder', deconv='transpose+conv', is_training=True):
    batch_size, _ = inputs.get_shape().as_list()

    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        with slim.arg_scope(
            [slim.conv2d_transpose],
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params={'is_training': is_training},
        ):
            with slim.arg_scope(
                [slim.conv2d],
                activation_fn=tf.nn.leaky_relu,
                normalizer_fn=None,
                normalizer_params={'is_training': is_training},
            ):
                net = slim.dropout(inputs)
                net = slim.fully_connected(inputs, 64 * 64 * conv_channel, activation_fn=tf.nn.leaky_relu)
                net = tf.reshape(net, [batch_size, 64, 64, conv_channel])

                net = slim.conv2d(net, conv_channel, [3, 3])
                net = slim.conv2d(net, conv_channel * 2, [3, 3])
                net = deconv_layer(net, 2, conv_channel, method=deconv)
                net = tf.nn.leaky_relu(net)

                net = slim.conv2d(net, conv_channel, [3, 3])
                net = slim.conv2d(net, conv_channel * 2, [3, 3])
                net = deconv_layer(net, 2, conv_channel, method=deconv)
                net = tf.nn.leaky_relu(net)

                net = slim.conv2d(net, conv_channel, [3, 3])
                net = slim.conv2d(net, out_channel, [3, 3])
    
    return net
