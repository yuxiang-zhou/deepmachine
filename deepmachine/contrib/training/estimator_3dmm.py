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


def format_folder(FLAGS):
    post_fix = 'lr{:02.1f}_d{:02.1f}_emb{:03d}_b{:02d}_3dmmw{:.3f}'.format(
        FLAGS.lr*1000, FLAGS.lr_decay *
        100, FLAGS.embedding, FLAGS.batch_size, FLAGS.m_weight
    )

    logdir = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
        FLAGS.logdir, post_fix
    )

    return logdir


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


def get_data_fn(FLAGS, N_VERTICES, INPUT_SHAPE=112):
    BATCH_SIZE = FLAGS.batch_size
    NUM_GPUS = len(FLAGS.gpu.split(','))

    def mesh_augmentation(meshes):
        # scale
        meshes *= tf.random_uniform([BATCH_SIZE, 1, 1]) * 0.7 + 0.8

        # translation
        meshes += tf.random_uniform([BATCH_SIZE, 1, 3]) * 0.4 - 0.2

        # rotation
        # pending

        return meshes

    def data_fn():

        keys_to_features = dm.utils.union_dict([
            dm.data.provider.features.image_feature(),
            dm.data.provider.features.matrix_feature('mesh'),
            dm.data.provider.features.matrix_feature('mesh/colour'),
            dm.data.provider.features.array_feature('mesh/mask'),
        ])

        dataset = tf.data.TFRecordDataset(
            FLAGS.dataset_path, num_parallel_reads=FLAGS.no_thread)

        # Shuffle the dataset
        dataset = dataset.shuffle(
            buffer_size=BATCH_SIZE * NUM_GPUS * FLAGS.no_thread)

        # Repeat the input indefinitly
        dataset = dataset.repeat()

        # Generate batches
        dataset = dataset.batch(BATCH_SIZE)

        # example proto decode
        def _parse_function(example_proto):

            parsed_features = tf.parse_example(example_proto, keys_to_features)
            feature_dict = {}

            # parse image
            def parse_single_image(feature):

                m = tf.image.decode_jpeg(feature, channels=3)
                m = tf.reshape(m, [256, 256, 3])
                m = tf.to_float(m) / 255.
                return m

            feature_dict['image'] = tf.image.resize_images(
                tf.map_fn(parse_single_image,
                          parsed_features['image'], dtype=tf.float32),
                [INPUT_SHAPE, INPUT_SHAPE]
            )

            # parse mesh
            m = tf.decode_raw(parsed_features['mesh'], tf.float32)
            m = tf.reshape(m, [-1, N_VERTICES, 3])
            feature_dict['mesh'] = mesh_augmentation(m)

            # parse mesh/colour
            m = tf.decode_raw(parsed_features['mesh/colour'], tf.float32)
            m = tf.reshape(m, [-1, N_VERTICES, 3])
            feature_dict['mesh/colour'] = m

            # create cmesh
            feature_dict['cmesh'] = tf.concat(
                [feature_dict['mesh'], feature_dict['mesh/colour']], axis=-1
            )

            # parse mask
            m = tf.decode_raw(parsed_features['mesh/mask'], tf.float32)
            m = tf.reshape(m, [-1, N_VERTICES, 1])
            feature_dict['mask'] = m

            return feature_dict, feature_dict

        # Parse the record into tensors.
        dataset = dataset.map(
            _parse_function, num_parallel_calls=FLAGS.no_thread)

        return dataset

    return data_fn


def get_model_fn(FLAGS, graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices, N_VERTICES, EPOCH_STEPS, TOTAL_EPOCH, FILTERS=[16, 32, 32, 64], EMBEDING=128, INPUT_SHAPE=112, inputs_channels=6):

    def model_fn(features, labels, mode, params):
        # define components

        # image encoder
        input_image = features['image']
        tf.summary.image('input', input_image)

        img_embedding = Encoder2D(input_image, EMBEDING)

        # decoder
        output_img_rec_mesh = MeshDecoder(
            img_embedding,
            inputs_channels,
            graph_laplacians,
            adj_matrices,
            upsamling_matrices,
            polynomial_order=6,
            filter_list=FILTERS
        )

        # Build estimator spec
        predictions = {
            'cmesh': output_img_rec_mesh
        }

        # PREDICT mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Build Additional Graph and Loss (for both TRAIN and EVAL modes)

        input_mesh = features['cmesh']
        mesh_embedding = MeshEncoder(
            input_mesh, EMBEDING, graph_laplacians, downsampling_matrices, filter_list=FILTERS)

        output_rec_mesh_ae = MeshDecoder(
            mesh_embedding,
            inputs_channels,
            graph_laplacians,
            adj_matrices,
            upsamling_matrices,
            polynomial_order=6,
            filter_list=FILTERS,
            reuse=True,
        )

        # define losses
        loss_3dmm = tf.reduce_mean(tf.losses.absolute_difference(
            labels['cmesh'], output_img_rec_mesh, weights=labels['mask']))
        loss_mesh_ae = tf.reduce_mean(tf.losses.absolute_difference(
            labels['cmesh'], output_rec_mesh_ae, weights=labels['mask']))
        total_loss = loss_3dmm + loss_mesh_ae
        tf.summary.scalar('loss/3dmm', loss_3dmm)
        tf.summary.scalar('loss/mesh_ae', loss_mesh_ae)
        tf.summary.scalar('loss/total', total_loss)

        global_steps = tf.train.get_global_step()

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:

            learning_rate = tf.train.exponential_decay(
                FLAGS.lr,
                global_steps,
                EPOCH_STEPS,
                0.98
            )

            tf.summary.scalar('lr', learning_rate)

            lr_3dmm = tf.train.piecewise_constant(
                global_steps, [int(EPOCH_STEPS*TOTAL_EPOCH / 2)], [1e-10, learning_rate])
            optimizer_3dmm = tf.train.AdamOptimizer(learning_rate=lr_3dmm)
            optimizer_3dmm = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer_3dmm, 5.0)
            train_op_3dmm = optimizer_3dmm.minimize(
                loss=loss_3dmm,
                global_step=global_steps,
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='image_encoder')
            )

            optimizer_mesh_ae = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
            optimizer_mesh_ae = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer_mesh_ae, 5.0)
            train_op_mesh_ae = optimizer_mesh_ae.minimize(
                loss=loss_mesh_ae,
                global_step=global_steps,
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mesh_encoder') +
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='mesh_decoder')
            )

            train_op = tf.group(train_op_3dmm, train_op_mesh_ae)

            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "mae": tf.metrics.mean_absolute_error(labels=labels['cmesh'], predictions=predictions["cmesh"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)

    return model_fn


def main():

    # flag definitions
    tf.app.flags.DEFINE_string(
        'meta_path', '/vol/atlas/homes/yz4009/databases/mesh/meta', '''path to meta files''')
    tf.app.flags.DEFINE_integer('embedding', 128, '''embedding''')
    tf.app.flags.DEFINE_float('m_weight', 1.0, '''embedding''')
    tf.app.flags.DEFINE_string(
        'pretrained_meshae', '/homes/yz4009/db/trained_model/mesh_ae/weights.h5', '''pretrained_meshae''')
    from deepmachine.flags import FLAGS

    # hyperparameters
    BATCH_SIZE = FLAGS.batch_size
    LR = FLAGS.lr
    EMBEDING = FLAGS.embedding
    LOGDIR = format_folder(FLAGS)
    N_VERTICES = 53215
    INPUT_SHAPE = 112
    STARTING_3DMM = -1 if FLAGS.pretrained_meshae else 100
    FILTERS = [16, 32, 32, 64]
    DB_SIZE = 40000
    NUM_GPUS = len(FLAGS.gpu.split(','))
    EPOCH_STEPS = DB_SIZE // (BATCH_SIZE * NUM_GPUS) * 2
    TOTAL_EPOCH = FLAGS.n_epoch

    # globel constant
    shape_model = mio.import_pickle(
        FLAGS.meta_path + '/all_all_all.pkl')
    trilist = shape_model.instance([]).trilist
    graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices = mio.import_pickle(
        FLAGS.meta_path + '/lsfm_LDUA.pkl', encoding='latin1')

    # configuration
    if NUM_GPUS > 1:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
    else:
        strategy = tf.contrib.distribute.OneDeviceStrategy(0)

    config = tf.estimator.RunConfig(
        train_distribute=strategy,
        save_checkpoints_steps=EPOCH_STEPS,
        save_summary_steps=100,
        keep_checkpoint_max=None,
    )

    # Set up Hooks
    class TimeHistory(tf.train.SessionRunHook):
        def begin(self):
            self._step = -1
            self.times = []
            self.total_epoch = TOTAL_EPOCH
            self.total_steps = EPOCH_STEPS * self.total_epoch

        def before_run(self, run_context):
            self._step += 1
            self.iter_time_start = time.time()

        def after_run(self, run_context, run_values):
            self.times.append(time.time() - self.iter_time_start)

            if self._step % 20 == 0:
                total_time = sum(self.times)
                avg_time_per_batch = np.mean(time_hist.times[-20:])
                estimate_finishing_time = (
                    self.total_steps - self._step) * avg_time_per_batch
                i_batch = self._step % EPOCH_STEPS
                i_epoch = self._step // EPOCH_STEPS

                
                print("INFO: Epoch [{}/{}], Batch [{}/{}]".format(i_epoch,self.total_epoch,i_batch,EPOCH_STEPS))
                print("INFO: Estimate Finishing time: {}".format(datetime.timedelta(seconds=estimate_finishing_time)))
                print("INFO: Image/sec: {}".format(BATCH_SIZE*NUM_GPUS/avg_time_per_batch))
                print("INFO: N GPUs: {}".format(NUM_GPUS))

    time_hist = TimeHistory()

    # Create the Estimator
    Fast_3DMM = tf.estimator.Estimator(
        model_fn=get_model_fn(
            FLAGS, graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices, N_VERTICES, EPOCH_STEPS, TOTAL_EPOCH, FILTERS=FILTERS, EMBEDING=EMBEDING, INPUT_SHAPE=INPUT_SHAPE
        ), model_dir=LOGDIR, config=config)

    # train
    Fast_3DMM.train(get_data_fn(
        FLAGS, N_VERTICES, INPUT_SHAPE=INPUT_SHAPE
    ), steps=EPOCH_STEPS * TOTAL_EPOCH, hooks=[time_hist])


if __name__ == '__main__':
    main()
