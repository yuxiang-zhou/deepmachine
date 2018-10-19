import tensorflow as tf
import numpy as np
from functools import partial
from keras.models import Model
from .. import layers
from ..base import K
from .module import conv2d, deconv2d, conv2dt, vae_sampling


def HGResiduleModule(inputs, out_channel, add_residule=True, kernel_initializer='glorot_uniform', **kwargs):
    net = inputs
    for ch, kernal in zip([out_channel // 2, out_channel // 2, out_channel], [1, 3, 1]):
        net = conv2d(
            net, ch, kernal,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer, **kwargs)

    if add_residule:
        res_net = conv2d(
            inputs, out_channel, 1,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer, **kwargs)
        net = layers.Add()([net, res_net])

    return net


def ResiduleModule(x, dim, ks=3, s=1, kernel_initializer='glorot_uniform', activation='relu', batch_norm='InstanceNormalization2D', **kwargs):
    y = conv2d(x, dim, ks, strides=s, padding='same', activation=activation,
               batch_norm=batch_norm, kernel_initializer=kernel_initializer, **kwargs)
    y = conv2d(y, dim, ks, strides=s, padding='same', activation=None,
               batch_norm=batch_norm, kernel_initializer=kernel_initializer, **kwargs)
    return layers.Add()([y, x])


def Encoding2D(inputs, out_channel, down_sample=True, module='Residule', kernel_initializer='glorot_uniform', **kwargs):
    module_fn = globals()['%sModule' % module] if type(
        module) is str else module

    net = module_fn(inputs, out_channel,
                    kernel_initializer=kernel_initializer,
                    **kwargs)

    if down_sample:
        net = conv2d(net, out_channel, 3, strides=2, padding='same',
                     kernel_initializer=kernel_initializer, **kwargs)

    return net


def Hourglass(inputs, output_shape, depth=4, nf=64, module='HGResidule', kernel_initializer='glorot_uniform', **kwargs):

    net = inputs
    net = conv2d(
        net, nf, 7,
        strides=2,
        padding='same',
        batch_norm=None,
        activation='relu',
        dropout=None,
        use_coordconv=kwargs['use_coordconv'] if 'use_coordconv' in kwargs else False,
        kernel_initializer=kernel_initializer)
    net = layers.MaxPool2D(pool_size=2)(net)
    # Down sampling
    skip_layers = []
    for s in range(depth):
        s = np.min([s, 3])
        skip_layers.append(net)
        net = Encoding2D(net, nf * 2 ** s, down_sample=True, activation='relu', kernel_initializer=kernel_initializer, module=module, **kwargs)

    # feature compression
    net = Encoding2D(
        net,
        nf * 2 ** depth,
        down_sample=False,
        activation='relu',
        kernel_initializer=kernel_initializer,
        module=module,
        **kwargs
    )

    # up sampling
    for s, s_layer in zip(range(depth), skip_layers[::-1]):
        s = (depth - s - 1)
        s = np.min([s, 3])
        net = deconv2d(
            net, nf * 2 ** s, 3,
            kernel_initializer=kernel_initializer, activation='relu', **kwargs)
        net = layers.Concatenate()([net, s_layer])

    # output regress
    net = deconv2d(
        net,
        nf,
        kernel_size=3,
        size=4,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_initializer, **kwargs)

    prediction = conv2d(
        net,
        output_shape[-1],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=kernel_initializer)

    return prediction


def Encoder2D(inputs, embedding, depth=2, nf=32, kernel_initializer='glorot_uniform', **kwargs):

    net = conv2d(
        inputs, nf, 3,
        activation='relu',
        kernel_initializer=kernel_initializer, **kwargs)

    for s in range(1, depth):
        s = np.min([s, 4])
        net = Encoding2D(net, nf * 2 ** s, module='HGResidule', **kwargs)

    s = np.min([depth, 4])
    net = Encoding2D(net, nf * 2 ** s, module='HGResidule',
                     down_sample=False, **kwargs)

    net = layers.Flatten()(net)
    net = layers.Dense(embedding)(net)
    net = layers.ReLU()(net)

    return net


def Decoder2D(inputs, out_shape, depth=2, nf=32, kernel_initializer='glorot_uniform', **kwargs):

    input_shape = np.array(list(np.array(
        out_shape[:-1]) / np.power([2, 2], depth)) + [inputs.shape.as_list()[-1]]).astype(np.int)

    net = layers.Dense(np.prod(input_shape))(inputs)
    net = layers.ReLU()(net)

    net = layers.Reshape(input_shape)(net)

    for s in range(depth):
        s = np.min([(depth - s - 1), 4])
        net = deconv2d(
            net, nf * 2 ** s, 3,
            activation='relu',
            kernel_initializer=kernel_initializer, **kwargs)

    net = conv2d(
        net,
        out_shape[-1], (3, 3),
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=kernel_initializer, **kwargs)

    return net


def AutoEncoder(inputs, output_shape, depth=2, embedding=128, nf=32, kernel_initializer='glorot_uniform', **kwargs):

    embedding = Encoder2D(inputs, embedding,
                          depth=depth, nf=nf, **kwargs)
    reconstruction = Decoder2D(
        embedding, output_shape, depth=depth, nf=nf, **kwargs)

    return reconstruction


def UNet(inputs, output_shape, nf=64, ks=4, **kwargs):
    ### UNet Definition ###
    # image is (256 x 256 x input_c_dim)
    e1 = conv2d(inputs, nf, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e1 is (128 x 128 x self.gf_dim)
    e2 = conv2d(e1, nf*2, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e2 is (64 x 64 x self.gf_dim*2)
    e3 = conv2d(e2, nf*4, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e3 is (32 x 32 x self.gf_dim*4)
    e4 = conv2d(e3, nf*8, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e4 is (16 x 16 x self.gf_dim*8)
    e5 = conv2d(e4, nf*8, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e5 is (8 x 8 x self.gf_dim*8)
    e6 = conv2d(e5, nf*8, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e6 is (4 x 4 x self.gf_dim*8)
    e7 = conv2d(e6, nf*8, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e7 is (2 x 2 x self.gf_dim*8)
    e8 = conv2d(e7, nf*8, ks, strides=2, padding='same',
                activation=True, batch_norm='InstanceNormalization2D')
    # e8 is (1 x 1 x self.gf_dim*8)

    d1 = deconv2d(e8, nf*8, ks, padding='same',
                  activation='relu', batch_norm='InstanceNormalization2D', dropout=0.5)
    d1 = layers.Concatenate()([d1, e7])
    # d1 is (2 x 2 x self.gf_dim*8*2)

    d2 = deconv2d(d1, nf*8, ks, padding='same',
                  activation='relu', batch_norm='InstanceNormalization2D', dropout=0.5)
    d2 = layers.Concatenate()([d2, e6])
    # d2 is (4 x 4 x self.gf_dim*8*2)

    d3 = deconv2d(d2, nf*8, ks, padding='same',
                  activation='relu', batch_norm='InstanceNormalization2D', dropout=0.5)
    d3 = layers.Concatenate()([d3, e5])
    # d3 is (8 x 8 x self.gf_dim*8*2)

    d4 = deconv2d(
        d3, nf*8, ks, padding='same', activation='relu', batch_norm='InstanceNormalization2D')
    d4 = layers.Concatenate()([d4, e4])
    # d4 is (16 x 16 x self.gf_dim*8*2)

    d5 = deconv2d(
        d4, nf*4, ks, padding='same', activation='relu', batch_norm='InstanceNormalization2D')
    d5 = layers.Concatenate()([d5, e3])
    # d5 is (32 x 32 x self.gf_dim*4*2)

    d6 = deconv2d(
        d5, nf*2, ks, padding='same', activation='relu', batch_norm='InstanceNormalization2D')
    d6 = layers.Concatenate()([d6, e2])
    # d6 is (64 x 64 x self.gf_dim*2*2)

    d7 = deconv2d(
        d6, nf, ks, padding='same', activation='relu', batch_norm='InstanceNormalization2D')
    d7 = layers.Concatenate()([d7, e1])
    # d7 is (128 x 128 x self.gf_dim*1*2)

    outputs = deconv2d(
        d7, 3, ks, padding='same', activation='tanh', batch_norm=None)
    # outputs is (256 x 256 x output_c_dim)
    return outputs


def Discriminator(inputs, nf=64, depth=4, ks=4, return_endpoints=False, **kwargs):
    net = inputs
    for i in range(depth):
        net = conv2d(
            net,
            nf * 2 ** i,
            ks,
            strides=2,
            padding='same',
            batch_norm='InstanceNormalization2D' if i > 0 else None,
            activation=True,
            **kwargs
        )

    validity = conv2d(
        net,
        1,
        1,
        strides=1,
        padding='same',
        batch_norm=False,
        activation=False,
    )

    if return_endpoints:
        return validity, net

    return validity


def ResNet50(inputs, output_shape, nf=64, n_residule=9, module='Residule', with_deconv=True, with_dense=False, embeding=256, n_classes=100, dropout=0.3, **kwargs):

    module_fn = globals()['%sModule' % module] if type(
        module) is str else module

    # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
    # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
    net = conv2d(inputs, nf, 7, strides=1, padding='same',
                 activation='relu', batch_norm='InstanceNormalization2D', **kwargs)
    net = conv2d(net, nf * 2, 3, strides=2, activation='relu',
                 batch_norm='InstanceNormalization2D', **kwargs)
    net = conv2d(net, nf * 4, 3, strides=2, activation='relu',
                 batch_norm='InstanceNormalization2D', **kwargs)
    # define G network with 9 resnet blocks
    for _ in range(n_residule):
        net = module_fn(net, nf*4, **kwargs)

    if with_deconv:
        net = deconv2d(net, nf*2, 3, size=2, activation='relu',
                       batch_norm='InstanceNormalization2D', **kwargs)
        net = deconv2d(net, nf, 3, size=2, activation='relu',
                       batch_norm='InstanceNormalization2D', **kwargs)
        net = conv2d(net, output_shape[-1], 7, strides=1,
                     padding='same', activation='tanh', batch_norm=None, **kwargs)
    elif with_dense:
        net = layers.Flatten()(net)
        net = layers.Dense(embeding, activation='relu')(net)
        net = layers.Dropput(dropout)(net)
        net = layers.Dense(n_classes, activation='softmax')(net)

    return net


def VAE(inputs, nf=32, ks=3, depth=2, embedding=64, latent=16, return_models=False, *args, **kwargs):
    # VAE model = encoder + decoder
    # build encoder model
    output_shape = K.int_shape(inputs)

    x = inputs
    for i in range(depth):
        x = conv2d(x, nf * (2**i), ks, strides=2,
                   activation='relu', *args, **kwargs)

    # shape info needed to build decoder model
    embedding_shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = layers.Flatten()(x)
    x = layers.Dense(embedding, activation='relu')(x)
    z_mean = layers.Dense(latent, name='z_mean')(x)
    z_log_var = layers.Dense(latent, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend

    z = layers.Lambda(vae_sampling, output_shape=(
        latent,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(
        inputs,
        [z_mean, z_log_var, z],
        name='encoder')

    # build decoder model
    latent_inputs = layers.Input(shape=(latent,), name='z_sampling')
    x = layers.Dense(embedding_shape[1] * embedding_shape[2]
                     * embedding_shape[3], activation='relu')(latent_inputs)
    x = layers.Reshape(
        (embedding_shape[1], embedding_shape[2], embedding_shape[3]))(x)

    for i in range(depth-1, -1, -1):
        x = deconv2d(x, nf*2**i, ks, strides=1, size=2,
                     activation='relu', *args, **kwargs)

    recon = conv2d(x, output_shape[-1], ks,
                   strides=1, activation=None, *args, **kwargs)

    # instantiate decoder model
    decoder = Model(latent_inputs, recon, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[-1])
    if return_models:
        return outputs, [encoder, decoder]
    return outputs
