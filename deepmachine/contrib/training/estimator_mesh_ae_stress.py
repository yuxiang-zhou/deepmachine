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
        print('    Relu ' + self.name + ':', n_channels)

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
        print('    Upsample ' + self.name + ':', Mp * M * Fin)
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
        L0, L1 = L.shape
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
            print('    Conv Sparse K_1 ' + self.name + ':', L0 * L1 * M)
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, nK):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            print('    Conv Sparse K_' + str(k) + ' ' + self.name + ':', L0 * L1 * M)
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [nK, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*nK])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable
        x = tf.matmul(x, W)  # N*M x Fout
        print('    Conv ' + self.name + ':', M * Fin*nK * Fout)
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

def MeshDecoder(inputs, out_channel, graph_laplacians, adj_matrices, upsamling_matrices, polynomial_order=6, filter_list=[16, 16, 16, 16], name='mesh_decoder', reuse=False, **kwargs):
    with tf.variable_scope(name, reuse=reuse):
        pool_size = list(map(lambda x: x.shape[0], adj_matrices))
        net = inputs
        net = tf.layers.Dense(pool_size[-1] * filter_list[-1])(net)
        print('    ' + net.name + ':', pool_size[-1] * filter_list[-1] * inputs.shape[-1])
        net = tf.reshape(net, [-1, pool_size[-1], filter_list[-1]])

        for i,(nf, nl, nu) in enumerate(zip(filter_list[::-1], graph_laplacians[-2::-1], upsamling_matrices[::-1])):
            net = MeshPoolTrans(nu)(net)
            net = MeshConv(
                nl, nf=nf, polynomial_order=polynomial_order, **kwargs)(net)
            net = MeshReLU1B()(net)

        net = MeshConv(graph_laplacians[0], nf=out_channel,
                       polynomial_order=polynomial_order, **kwargs)(net)

    return net

def get_config():

    # flag definitions
    tf.app.flags.DEFINE_string(
        'meta_path', '/vol/phoebe/yz4009/databases/mesh/meta', '''path to meta files''')
    tf.app.flags.DEFINE_string(
        'ldua_filename', 'coma_LDUA.pkl', '''name of LDUA file name''')
    tf.app.flags.DEFINE_string(
        'test_path', '', '''path to test files''')
    tf.app.flags.DEFINE_integer('embedding', 128, '''embedding''')
    tf.app.flags.DEFINE_integer('k_poly', 6, '''embedding''')
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
            self.NK = FLAGS.k_poly
            self.INPUT_SHAPE = 112
            self.FILTERS = [16, 16, 16, 16]
            self.DB_SIZE = 10000
            self.NUM_GPUS = len(FLAGS.gpu.split(','))
            self.EPOCH_STEPS = self.DB_SIZE // (self.BATCH_SIZE * self.NUM_GPUS)
            self.TOTAL_EPOCH = FLAGS.n_epoch
            self.no_thread = FLAGS.no_thread
            self.dataset_path = FLAGS.dataset_path
            self.test_path = FLAGS.test_path
            
            # globel constant
            self.graph_laplacians, self.downsampling_matrices, self.upsamling_matrices, self.adj_matrices = mio.import_pickle(
                os.path.join(FLAGS.meta_path, FLAGS.ldua_filename), encoding='latin1')
            self.N_VERTICES = self.graph_laplacians[0].shape[0]


    return Config()

def main():

    config = get_config()
    # configuration
    tf.reset_default_graph()
    mesh_embedding = tf.constant(np.random.sample([config.BATCH_SIZE, config.EMBEDING]), dtype=tf.float32)
    # mesh_embedding = tf.keras.layers.Input(shape=[config.EMBEDING])
    # decoder
    print('Operation Count:')
    output_rec_mesh = MeshDecoder(
        mesh_embedding,
        config.inputs_channels,
        config.graph_laplacians,
        config.adj_matrices,
        config.upsamling_matrices,
        polynomial_order=config.NK,
        filter_list=config.FILTERS
    )
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    num_iteration = 100
    
    total_time = []
    for i in range(num_iteration):
        star_time = time.time()
        output = sess.run(output_rec_mesh)
        end_time = time.time()
        single_iteration = (end_time - star_time) * 1000
        total_time.append(single_iteration)
    min_speed = np.min(total_time) / config.BATCH_SIZE
    print('Single Inference Speed: {:.5f} ms'.format(min_speed))
    print('FPS: {:.5f}'.format(1000 / min_speed))
    print(output.shape)
    
    
    print('Parameter Summary: ')
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        print(variable.name, shape, variable_parameters)
    print(total_parameters)

if __name__ == '__main__':
    main()
