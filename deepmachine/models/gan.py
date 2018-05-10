import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from .stackedHG import hourglass_arg_scope_tf, bottleneck_module, deconv_layer, hourglass

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


    

def batchnorm(input, name='batchnorm'):
    with tf.variable_scope(name):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

batch_norm = batchnorm

def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


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


def lrelu(x, a=0.2, name="lrelu"):
    with tf.name_scope(name):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


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


def generator(inputs, output_dim, deconv='transpose+bn', bottleneck='bottleneck', reuse=False, name="generator"):

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

def generator_hourglass_module(inputs, depth=0, deconv='transpose_bn', bottleneck='bottleneck'):

    bm_fn = globals()['%s_module' % bottleneck]

    with tf.variable_scope('depth_{}'.format(depth)):
        # buttom up layers
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu):
            net = slim.max_pool2d(inputs, [2, 2], scope='pool')
            net = slim.stack(net, bm_fn, [
                            (256, None), (256, None), (256, None)], scope='buttom_up')

        # connecting layers
        if depth > 0:
            net = generator_hourglass_module(net, depth=depth - 1, deconv=deconv)
        else:
            net = bm_fn(
                net, out_channel=512, res=512, scope='connecting')

        # top down layers
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            net = bm_fn(net, out_channel=512,
                        res=512, scope='top_down')
            net = deconv_layer(net, 2, 512, method=deconv)
        # residual layers
        net += slim.stack(inputs, bm_fn,
                          [(256, None), (256, None), (512, 512)], scope='res')

        return net

def generator_hourglass(
    inputs,
    output_channels=3,
    n_encoder=5,
    deconv='transpose+bn',
    bottleneck='bottleneck',
    reuse=False,
    name='generator'
):
    """Defines a lightweight resnet based model for dense estimation tasks.
    Args:
      inputs: A `Tensor` with dimensions [num_batches, height, width, depth].
      scale: A scalar which denotes the factor to subsample the current image.
      output_channels: The number of output channels. E.g., for human pose
        estimation this equals 13 channels.
    Returns:
      A `Tensor` of dimensions [num_batches, height, width, output_channels]."""

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        with slim.arg_scope(hourglass_arg_scope_tf()):
            with slim.arg_scope([slim.conv2d], activation_fn=lrelu):
                # D1
                net = slim.conv2d(inputs, 64, (7, 7), 2, scope='conv1', normalizer_fn=None)
                net = bottleneck_module(net, out_channel=128,
                                        res=128, scope='bottleneck1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                # D2
                net = slim.stack(net, bottleneck_module, [
                                (128, None), (128, None), (256, 256)], scope='conv2')

            # hourglasses (D3,D4,D5)
            with tf.variable_scope('hourglass'):
                net = generator_hourglass_module(
                    net, depth=n_encoder, deconv=deconv, bottleneck=bottleneck)

            # final layers (D6, D7)
            net = slim.stack(net, slim.conv2d, [(512, [1, 1]), (256, [1, 1]),
                                                (output_channels, [1, 1])
                                                ], scope='conv3')

            net = deconv_layer(net, 4, output_channels, method=deconv)

        out_layer = tf.nn.tanh(net)

        return out_layer
    
    
    
def create_generator(generator_inputs, generator_outputs_channels, reuse=False, name='generator', ngf=64):
    with tf.variable_scope(name):
        # inputs is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
    
    
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv(generator_inputs, ngf, stride=2)
            layers.append(output)

        layer_specs = [
            ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2)
                output = batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(rectified, out_channels)
                output = batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]
    
    
def create_discriminator(discrim_inputs, discrim_targets, reuse=False, name='discriminator', ndf=64):
    with tf.variable_scope(name):
        # inputs is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
            
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        cond_input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(cond_input, ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2**(n_layers-i-1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]