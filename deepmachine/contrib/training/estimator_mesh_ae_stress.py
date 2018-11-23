# basic library
import os
import shutil
import math
import time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
import h5py
import pandas as pd
import datetime

from menpo.shape import ColouredTriMesh, PointCloud
from menpo.image import Image
from menpo.transform import Homogeneous
from menpo3d.rasterize import rasterize_mesh
from pathlib import Path
from functools import partial
from itwmm.visualize import lambertian_shading


# deepmachine
import scipy
import tensorflow as tf
import deepmachine as dm
from deepmachine.utils import mesh as graph
from deepmachine.layers.mesh_renderer.mesh_renderer import mesh_renderer

tf.logging.set_verbosity(tf.logging.INFO)


# Custom Layer

class MeshReLU1B(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self.bias = self.add_variable(
            name='kernel',
            shape=[1, n_channels]
        )

    def call(self, x):
        return tf.nn.relu(x + self.bias)


class MeshReLU2B(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""
        _, n_vertexes, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self.bias = self.add_variable(
            name='kernel',
            shape=[n_vertexes, n_channels])

    def call(self, x):
        return tf.nn.relu(x + self.bias)


class MeshPoolTrans(tf.keras.layers.Layer):

    def poolwT(self, x):
        L = self._gl
        Mp = L.shape[0]
        _, M, Fin = x.get_shape().as_list()
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, -1])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, -1])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

        return x

    def __init__(self, graph_laplacians, **kwargs):
        self._gl = graph_laplacians.astype(np.float32)
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.poolwT(x)


MeshPool = MeshPoolTrans


class MeshConv(tf.keras.layers.Layer):

    def chebyshev5(self, x, L, Fout, nK):
        L = L.astype(np.float32)
        _, M, Fin = x.get_shape().as_list()
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if nK > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, nK):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [nK, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*nK])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable
        x = tf.matmul(x, W)  # N*M x Fout
        out = tf.reshape(x, [-1, M, Fout])  # N x M x Fout
        return out

    def __init__(self, graph_laplacians, polynomial_order=6, nf=16, **kwargs):
        self._gl = graph_laplacians
        self._nf = nf
        self._po = polynomial_order
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Bias and ReLU. One bias per filter."""

        _, _, n_channels = input_shape.as_list()
        # Create a trainable weight variable for this layer.
        self._weight_variable = self.add_variable(
            name='kernel',
            shape=[n_channels * self._po, self._nf]
        )

    def call(self, x):
        return self.chebyshev5(x, self._gl, self._nf, self._po)


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

            net = MeshConv(
                nl, nf=nf, polynomial_order=polynomial_order, **kwargs)(net)
            net = MeshReLU1B()(net)
            net = MeshPool(nd)(net)

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
            net = MeshPoolTrans(nu)(net)
            net = MeshConv(
                nl, nf=nf, polynomial_order=polynomial_order, **kwargs)(net)
            net = MeshReLU1B()(net)

        net = MeshConv(graph_laplacians[0], nf=out_channel,
                       polynomial_order=polynomial_order, **kwargs)(net)

    return net


# Component Definition

def get_data_fn(config, dataset_path, is_training=True, new_format=False):
    BATCH_SIZE = config.BATCH_SIZE
    NUM_GPUS = config.NUM_GPUS

    def mesh_augmentation(meshes):
        # scale
        meshes *= tf.random_uniform([BATCH_SIZE, 1, 1]) * 0.7 + 0.8

        # translation
        meshes += tf.random_uniform([BATCH_SIZE, 1, 3]) * 0.4 - 0.2

        # rotation
        # pending

        return meshes

    def gen():
        for i in range(10000):
            yield i

    def data_fn():
        dataset = tf.data.Dataset.from_generator(
            gen, tf.int64, tf.TensorShape([])
        )

        # Shuffle the dataset
        dataset = dataset.shuffle(
            buffer_size=BATCH_SIZE * NUM_GPUS * config.no_thread)

        # Repeat the input indefinitly
        dataset = dataset.repeat()

        # Generate batches
        dataset = dataset.batch(BATCH_SIZE)

        # example proto decode
        def _parse_function(data):
            data_dict = {
                'cmesh': tf.constant(np.ones([BATCH_SIZE, config.N_VERTICES, 6]), dtype=tf.float32),
                'mask': tf.constant(np.ones([BATCH_SIZE, config.N_VERTICES, 1]), dtype=tf.float32)
            }

            return data_dict, data_dict
            
        # Parse the record into tensors.
        dataset = dataset.map(
            _parse_function, num_parallel_calls=config.no_thread)


        return dataset

    return data_fn 


def get_model_fn(config):
    BATCH_SIZE = config.BATCH_SIZE
    def model_fn(features, labels, mode, params):
        # define components

        # mesh encoder
        input_mesh = features['cmesh']
        mesh_embedding = MeshEncoder(
            input_mesh, config.EMBEDING, config.graph_laplacians, config.downsampling_matrices, filter_list=config.FILTERS)

        # decoder
        output_rec_mesh = MeshDecoder(
            mesh_embedding,
            config.inputs_channels,
            config.graph_laplacians,
            config.adj_matrices,
            config.upsamling_matrices,
            polynomial_order=6,
            filter_list=config.FILTERS
        )

        # PREDICT mode
        # Build estimator spec
        predictions = {
            'cmesh': output_rec_mesh
        }
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # define losses
        total_loss = tf.reduce_mean(tf.losses.absolute_difference(
            labels['cmesh'], output_rec_mesh, weights=labels['mask']
        ))

        tf.summary.scalar('loss/total', total_loss)
        global_steps = tf.train.get_global_step()

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            

            learning_rate = tf.train.exponential_decay(
                config.LR,
                global_steps,
                config.EPOCH_STEPS,
                config.lr_decay
            )

            tf.summary.scalar('lr', learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
            train_op = optimizer.minimize(
                loss=total_loss,
                global_step=global_steps,
            )

            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "Reconstruction_ALL": tf.metrics.mean_absolute_error(labels=labels['cmesh'], predictions=predictions["cmesh"]),
            "Reconstruction_MESH": tf.metrics.mean_absolute_error(labels=labels['cmesh'][...,:3], predictions=predictions["cmesh"][...,:3]),
            "Reconstruction_COLOUR": tf.metrics.mean_absolute_error(labels=labels['cmesh'][...,3:], predictions=predictions["cmesh"][...,3:]),
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)

    return model_fn


def get_config():

    # flag definitions
    tf.app.flags.DEFINE_string(
        'meta_path', '/vol/atlas/homes/yz4009/databases/mesh/meta', '''path to meta files''')
    tf.app.flags.DEFINE_string(
        'test_path', '', '''path to test files''')
    tf.app.flags.DEFINE_integer('embedding', 128, '''embedding''')
    from deepmachine.flags import FLAGS

    class Config:

        def format_folder(self, FLAGS):
            post_fix = 'lr{:02.1f}_d{:02.1f}_emb{:03d}_b{:02d}'.format(
                FLAGS.lr, FLAGS.lr_decay, FLAGS.embedding, FLAGS.batch_size
            )

            logdir = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
                FLAGS.logdir, post_fix
            )

            return logdir

        def __init__(self, *args, **kwargs):
            # hyperparameters
            self.BATCH_SIZE = FLAGS.batch_size
            self.LR = FLAGS.lr
            self.inputs_channels = 6
            self.lr_decay = FLAGS.lr_decay
            self.EMBEDING = FLAGS.embedding
            self.LOGDIR = self.format_folder(FLAGS)
            self.N_VERTICES = 28431
            self.INPUT_SHAPE = 112
            self.FILTERS = [16, 32, 32, 64]
            self.DB_SIZE = 10000
            self.NUM_GPUS = len(FLAGS.gpu.split(','))
            self.EPOCH_STEPS = self.DB_SIZE // (self.BATCH_SIZE * self.NUM_GPUS)
            self.TOTAL_EPOCH = FLAGS.n_epoch
            self.no_thread = FLAGS.no_thread
            self.dataset_path = FLAGS.dataset_path
            self.test_path = FLAGS.test_path
            
            # globel constant
            self.graph_laplacians, self.downsampling_matrices, self.upsamling_matrices, self.adj_matrices = mio.import_pickle(
                FLAGS.meta_path + '/mein3dcrop_LDUA.pkl', encoding='latin1')


    return Config()

def main():

    config = get_config()
    # configuration
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=config.NUM_GPUS) if config.NUM_GPUS > 1 else None
    config_estimator = tf.estimator.RunConfig(
        train_distribute=strategy,
        save_checkpoints_steps=config.EPOCH_STEPS,
        save_summary_steps=100,
        keep_checkpoint_max=None,
    )

    # Create the Estimator
    Fast_3DMM = tf.estimator.Estimator(
        model_fn=get_model_fn(config), model_dir=config.LOGDIR, config=config_estimator)

    

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            "cmesh": np.random.sample([2, config.N_VERTICES, 6]).astype(np.float32),
        }, shuffle=False, batch_size=1
    )

    total_time = 0
    for i in range(1):
        result = Fast_3DMM.predict(input_fn=input_fn)

        # timing
        #warm up
        for idx, r in enumerate(result):
            if idx == 0:
                star_time = time.time()
        
        end_time = time.time()
        single_iteration = (end_time - star_time) * 1000
        total_time += single_iteration
    print('Speed: {:.5f} ms'.format(total_time / 64))

if __name__ == '__main__':
    main()
