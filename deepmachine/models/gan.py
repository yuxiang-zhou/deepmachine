import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from .stackedHG import hourglass, hourglass_arg_scope_tf


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(
                               stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(
                                         stddev=stddev),
                                     biases_initializer=None)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def discriminator(inputs, output_dim=64, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # inputs is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(inputs, output_dim, name='d_h0_conv'))

        # h0 is (128 x 128 x self.output_dim)
        h1 = lrelu(batch_norm(
            conv2d(h0, output_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.output_dim*2)
        h2 = lrelu(batch_norm(
            conv2d(h1, output_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.output_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, output_dim *
                                     8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.output_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)

        return h4


def generator(inputs, output_dim, deconv='transpose+conv', bottleneck='bottleneck_inception', reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        net, _ = hourglass(inputs,
                        scale=1,
                        regression_channels=output_dim,
                        classification_channels=0,
                        deconv=deconv,
                        bottleneck=bottleneck)

        return net


gan_arg_scope_tf = hourglass_arg_scope_tf


def generator_resnet(image, output_dim, gf_dim=32, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID',
                                  name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [
                       [0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, ks, s, padding='VALID',
                                  name=name + '_c2'), name + '_bn2')
            return y + x

        s = 256
        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(batch_norm(
            conv2d(c0, gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(batch_norm(
            conv2d(c1, gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(batch_norm(
            conv2d(c2, gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, gf_dim * 4, name='g_r1')
        r2 = residule_block(r1, gf_dim * 4, name='g_r2')
        r3 = residule_block(r2, gf_dim * 4, name='g_r3')
        r4 = residule_block(r3, gf_dim * 4, name='g_r4')
        r5 = residule_block(r4, gf_dim * 4, name='g_r5')
        r6 = residule_block(r5, gf_dim * 4, name='g_r6')
        r7 = residule_block(r6, gf_dim * 4, name='g_r7')
        r8 = residule_block(r7, gf_dim * 4, name='g_r8')
        r9 = residule_block(r8, gf_dim * 4, name='g_r9')

        d1 = deconv2d(r9, gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(batch_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(batch_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv2d(d2, output_dim, 7, 1, padding='VALID', name='g_pred_c')
        pred = tf.nn.tanh(batch_norm(pred, 'g_pred_bn'))

        return pred
