import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from .stackedHG import hourglass_arg_scope_tf, bottleneck_module, deconv_layer, hourglass

def encoder(inputs, out_channel=1024, reuse=False, scope='encoder'):
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
            stride=2
        ):

            net = slim.conv2d(inputs, out_channel // 8, [5, 5])
            net = slim.conv2d(net, out_channel // 4, [5, 5])
            net = slim.conv2d(net, out_channel // 2, [5, 5])
            net = slim.conv2d(net, out_channel, [5, 5])
            # net = slim.flatten(net)
            # net = slim.fully_connected(net, out_channel, activation_fn=tf.nn.leaky_relu)
            # net = slim.fully_connected(net, 32 * 32 * out_channel, activation_fn=tf.nn.leaky_relu)
            # net = tf.reshape(net, [batch_size, 32, 32, out_channel])

            tf.summary.image(
                'encoder',
                tf.reduce_mean(tf.sigmoid(net), axis=-1)[..., None],
                max_outputs=3)
    
    return net

def decoder(inputs, out_channel=3, reuse=False, scope='decoder'):
    _, _, _, in_channel = inputs.get_shape().as_list()
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        with slim.arg_scope(
            [slim.conv2d_transpose],
            activation_fn=None,
            normalizer_fn=None
        ):
            net = deconv_layer(inputs, 2, in_channel // 2, method='transpose')
            net = tf.nn.leaky_relu(net)
            net = deconv_layer(net, 2, in_channel // 4, method='transpose')
            net = tf.nn.leaky_relu(net)
            net = deconv_layer(net, 2, in_channel // 8, method='transpose')
            net = tf.nn.leaky_relu(net)
            net = deconv_layer(net, 2, out_channel, method='transpose')
            net = tf.sigmoid(net)
    
    return net
