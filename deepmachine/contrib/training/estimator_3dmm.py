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

# Component Definition

def get_data_fn(config, dataset_path, is_training=True, format='tf'):
    BATCH_SIZE = config.BATCH_SIZE
    NUM_GPUS = config.NUM_GPUS

    def data_fn_tf():

        keys_to_features = dm.utils.union_dict([
            dm.data.provider.features.image_feature(),
            dm.data.provider.features.matrix_feature('mesh'),
            dm.data.provider.features.matrix_feature('mesh/in_img'),
            dm.data.provider.features.matrix_feature('mesh/colour'),
            dm.data.provider.features.array_feature('mesh/mask'),
        ])

        dataset = tf.data.TFRecordDataset(
            dataset_path, num_parallel_reads=config.no_thread)

        # Shuffle the dataset
        dataset = dataset.shuffle(
            buffer_size=BATCH_SIZE * NUM_GPUS * config.no_thread)

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
                [config.INPUT_SHAPE, config.INPUT_SHAPE]
            )

            # parse mesh
            m = tf.decode_raw(parsed_features['mesh'], tf.float32)
            m = tf.to_float(tf.reshape(m, [-1, config.N_VERTICES, 3]))
            feature_dict['mesh'] = m

            # parse mesh in image
            m = tf.decode_raw(parsed_features['mesh/in_img'], tf.float32)
            m = tf.to_float(tf.reshape(m, [-1, config.N_VERTICES, 3]))
            feature_dict['mesh/in_img'] = m

            # parse mesh/colour
            m = tf.decode_raw(parsed_features['mesh/colour'], tf.float32)
            m = tf.reshape(m, [-1, config.N_VERTICES, 3])
            feature_dict['mesh/colour'] = tf.to_float(m)

            # create cmesh
            feature_dict['cmesh'] = tf.concat(
                [feature_dict['mesh'], feature_dict['mesh/colour']], axis=-1
            )

            # parse mask
            m = tf.decode_raw(parsed_features['mesh/mask'], tf.float32)
            m = tf.reshape(m, [-1, config.N_VERTICES, 1])
            feature_dict['mask'] = tf.to_float(m)
            

            return feature_dict, feature_dict

        # Parse the record into tensors.
        dataset = dataset.map(
            _parse_function, num_parallel_calls=config.no_thread)

        return dataset

    return data_fn_tf

def get_model_fn(config):
    BATCH_SIZE = config.BATCH_SIZE
    

    def model_fn(features, labels, mode, params):

        input_image = features['image']

        # image encoder
        encoder_embedding = dm.networks.tfl.Encoder2D(input_image, config.EMBEDING)

        # decoder
        output_rec_cmesh = dm.networks.tfl.MeshDecoder(
            encoder_embedding,
            config.inputs_channels,
            config.graph_laplacians,
            config.adj_matrices,
            config.upsamling_matrices,
            polynomial_order=6,
            filter_list=config.FILTERS
        )

        with tf.variable_scope('projection'):

            output_mesh_proj = dm.layers.tfl.MeshConv(config.graph_laplacians[0], nf=3, name='proj_conv')(output_rec_cmesh[..., :3])
            output_mesh_proj = tf.layers.batch_normalization(output_mesh_proj)
            output_mesh_proj = dm.layers.tfl.MeshReLU1B(name='proj_relu1b')(output_mesh_proj)

        # PREDICT mode
        # Build estimator spec
        predictions = {
            'cmesh': output_rec_cmesh,
            'pcmesh': tf.concat([output_mesh_proj, output_rec_cmesh[...,3:]], axis=-1)
        }
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


        # Build Additional Graph and Loss (for both TRAIN and EVAL modes)
        ## rendering config
        eye = tf.constant(BATCH_SIZE * [[0.0, 0.0, -2.0]], dtype=tf.float32)
        center = tf.constant(BATCH_SIZE * [[0.0, 0.0, 0.0]], dtype=tf.float32)
        world_up = tf.constant(BATCH_SIZE * [[-1.0, 0.0, 0.0]], dtype=tf.float32)
        ambient_colors = tf.constant(BATCH_SIZE * [[1., 1., 1.]], dtype=tf.float32) * 0.1
        light_positions = tf.constant(BATCH_SIZE * [[[2.0, 0, -2.0], [-2.0, 0, -2.0]]])
        light_intensities = tf.ones([BATCH_SIZE, 1, 3], dtype=tf.float32)
        mesh_triangles = tf.constant(config.trilist, dtype=tf.int32)

        def render_fn(cmesh):
            
            ## predicted mesh render
            mesh_v = cmesh[...,:3]
            mesh_c = cmesh[...,3:]
            mesh_c = tf.clip_by_value(mesh_c,0,1)
            mesh_v.set_shape([BATCH_SIZE, config.N_VERTICES, 3])
            mesh_n = tf.nn.l2_normalize(mesh_v, axis=2)
            mesh_n.set_shape([BATCH_SIZE, config.N_VERTICES, 3])

            mesh_r = mesh_renderer(
                mesh_v,
                triangles=mesh_triangles,
                normals=mesh_n,
                diffuse_colors=mesh_c,
                camera_position=eye,
                camera_lookat=center,
                camera_up=world_up,
                light_positions=light_positions,
                light_intensities=light_intensities,
                image_width=config.INPUT_SHAPE,
                image_height=config.INPUT_SHAPE,
                specular_colors=None,
                shininess_coefficients=None,
                ambient_color=ambient_colors,
                model_transform=None,
                fov_y=40.0,
                near_clip=0.01,
                far_clip=10.0
            )

            return mesh_r

        # mesh encoder
        input_mesh = features['cmesh']

        mesh_embedding = dm.networks.tfl.MeshEncoder(
            input_mesh, config.EMBEDING, config.graph_laplacians, config.downsampling_matrices, filter_list=config.FILTERS)

        output_rec_cmesh_ae = dm.networks.tfl.MeshDecoder(
            mesh_embedding,
            config.inputs_channels,
            config.graph_laplacians,
            config.adj_matrices,
            config.upsamling_matrices,
            polynomial_order=6,
            filter_list=config.FILTERS,
            reuse=True,
        )

        # in-graph rendering
        gt_mesh_r = render_fn(input_mesh)
        c3dmm_mesh_r = render_fn(output_rec_cmesh)
        ae_mesh_r = render_fn(output_rec_cmesh_ae)
        
        ## summaries
        tf.summary.image('image/input', input_image)
        tf.summary.image('image/mesh/gt', gt_mesh_r)
        tf.summary.image('image/mesh/pred', c3dmm_mesh_r)
        tf.summary.image('image/mesh/ae', ae_mesh_r)

        # define losses
        ## Autoencoder Losses
        loss_shape = tf.reduce_mean(tf.losses.absolute_difference(
            features['cmesh'][...,:3], output_rec_cmesh_ae[...,:3]
        ))

        loss_appearance = tf.reduce_mean(tf.losses.absolute_difference(
            features['cmesh'][...,3:], output_rec_cmesh_ae[...,3:], weights=labels['mask']
        ))
        
        canny_weight = dm.utils.tf_canny(gt_mesh_r)
        tf.summary.image('image/render/weight', canny_weight)
        loss_render = tf.reduce_mean(tf.losses.absolute_difference(
            gt_mesh_r, ae_mesh_r, weights=canny_weight
        ))
        

        tf.summary.scalar('loss/ae/shape', loss_shape)
        tf.summary.scalar('loss/ae/appearance', loss_appearance)
        tf.summary.scalar('loss/ae/render', loss_render)

        loss_ae = loss_shape + 0.5*loss_appearance + 0.5*loss_render
        loss_ae /= 2.

        tf.summary.scalar('loss/ae/total', loss_ae)

        ## 3DMM Losses

        loss_shape = tf.reduce_mean(tf.losses.absolute_difference(
            features['cmesh'][...,:3], output_rec_cmesh[...,:3]
        ))

        loss_appearance = tf.reduce_mean(tf.losses.absolute_difference(
            features['cmesh'][...,3:], output_rec_cmesh[...,3:], weights=labels['mask']
        ))
        
        canny_weight = dm.utils.tf_canny(gt_mesh_r)
        tf.summary.image('image/render/weight', canny_weight)
        loss_render = tf.reduce_mean(tf.losses.absolute_difference(
            gt_mesh_r, c3dmm_mesh_r, weights=canny_weight
        ))

        loss_projection = tf.reduce_mean(tf.losses.absolute_difference(
            features['mesh/in_img'], output_mesh_proj
        ))

        tf.summary.scalar('loss/3dmm/shape', loss_shape)
        tf.summary.scalar('loss/3dmm/appearance', loss_appearance)
        tf.summary.scalar('loss/3dmm/render', loss_render)
        tf.summary.scalar('loss/3dmm/projection', loss_projection)

        loss_3dmm = loss_shape + 0.5*loss_appearance + 0.5*loss_render + loss_projection
        loss_3dmm /= 3.

        tf.summary.scalar('loss/3dmm/total', loss_3dmm)

        ## total loss
        loss_total = loss_ae + loss_3dmm
        tf.summary.scalar('loss/total', loss_3dmm)

        ## global steps
        global_steps = tf.train.get_global_step()

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:

            learning_rate = tf.train.exponential_decay(
                config.LR,
                global_steps,
                config.EPOCH_STEPS,
                config.lr_decay
            )

            tf.summary.scalar('lr/ae', learning_rate)

            ## 3dmm train op
            tf.summary.scalar('lr/3dmm', learning_rate)
            optimizer_3dmm = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            optimizer_3dmm = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer_3dmm, 5.0)
            train_op_3dmm = optimizer_3dmm.minimize(
                loss=loss_3dmm,
                global_step=global_steps,
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='image_encoder'
                ) + tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='projection'
                )
            )

            ## autoencoder train op
            optimizer_mesh_ae = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate / 100.)
            optimizer_mesh_ae = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer_mesh_ae, 5.0)
            train_op_mesh_ae = optimizer_mesh_ae.minimize(
                loss=loss_ae,
                global_step=global_steps,
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mesh_encoder') +
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='mesh_decoder')
            )

            train_op = tf.group(train_op_3dmm, train_op_mesh_ae)

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_total, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "Reconstruction/All": tf.metrics.mean_absolute_error(labels=features['cmesh'], predictions=predictions["cmesh"]),
            "Reconstruction/Mesh": tf.metrics.mean_absolute_error(labels=features['cmesh'][...,:3], predictions=predictions["cmesh"][...,:3]),
            "Reconstruction/Appearance": tf.metrics.mean_absolute_error(labels=features['cmesh'][...,3:], predictions=predictions["cmesh"][...,3:]),
            "Reconstruction/Proj": tf.metrics.mean_absolute_error(labels=features['mesh/in_img'], predictions=output_mesh_proj),
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_total, eval_metric_ops=eval_metric_ops)

    return model_fn


def get_config():

    # flag definitions
    tf.app.flags.DEFINE_string(
        'meta_path', '/vol/atlas/homes/yz4009/databases/mesh/meta', '''path to meta files''')
    tf.app.flags.DEFINE_string(
        'test_path', '', '''path to test files''')
    tf.app.flags.DEFINE_string(
        'warm_start_from', None, '''path to test files''')
    tf.app.flags.DEFINE_integer('embedding', 128, '''embedding''')
    from deepmachine.flags import FLAGS

    class Config:

        def format_folder(self, FLAGS):
            post_fix = 'lr{:f}_d{:f}_emb{:03d}_b{:02d}'.format(
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
            self.warm_start_from = FLAGS.warm_start_from
            self.EMBEDING = FLAGS.embedding
            self.LOGDIR = self.format_folder(FLAGS)
            # self.N_VERTICES = 28431
            self.N_VERTICES = 53215
            self.INPUT_SHAPE = 112
            self.FILTERS = [16, 32, 32, 64]
            self.DB_SIZE = 30000
            self.NUM_GPUS = len(FLAGS.gpu.split(','))
            self.EPOCH_STEPS = self.DB_SIZE // (self.BATCH_SIZE * self.NUM_GPUS) * 2
            self.TOTAL_EPOCH = FLAGS.n_epoch
            self.no_thread = FLAGS.no_thread
            self.dataset_path = FLAGS.dataset_path
            self.test_path = FLAGS.test_path
            self.train_format = 'tf' if FLAGS.dataset_path.endswith('.tfrecord') else 'h5py'
            self.test_format = 'tf' if FLAGS.test_path.endswith('.tfrecord') else 'h5py'
            self.luda_file = 'lsfm_LDUA.pkl'
            self.symetric_index = np.load(FLAGS.meta_path + '/symetric_index_full.npy').astype(np.int32)
            # globel constant
            self.shape_model = mio.import_pickle(FLAGS.meta_path + '/all_all_all.pkl', encoding='latin1')
            self.trilist = self.shape_model.instance([]).trilist
            self.graph_laplacians, self.downsampling_matrices, self.upsamling_matrices, self.adj_matrices = mio.import_pickle(
                FLAGS.meta_path + '/' + self.luda_file, encoding='latin1')


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

    # Set up Hooks
    class TimeHistory(tf.train.SessionRunHook):

        def begin(self):
            self._step_tf = tf.train.get_global_step()
            self.times = []
            self.total_epoch = config.TOTAL_EPOCH
            self.total_steps = config.EPOCH_STEPS * self.total_epoch

        def before_run(self, run_context):
            self.iter_time_start = time.time()

        def after_run(self, run_context, run_values):
            self._step = run_context.session.run(self._step_tf)
            self.times.append(time.time() - self.iter_time_start)

            if self._step % 20 == 0:
                total_time = sum(self.times)
                avg_time_per_batch = np.mean(time_hist.times[-20:])
                estimate_finishing_time = (
                    self.total_steps - self._step) * avg_time_per_batch
                i_batch = self._step % config.EPOCH_STEPS
                i_epoch = self._step // config.EPOCH_STEPS

                
                print("INFO: Epoch [{}/{}], Batch [{}/{}]".format(i_epoch,self.total_epoch,i_batch,config.EPOCH_STEPS))
                print("INFO: Estimate Finishing time: {}".format(datetime.timedelta(seconds=estimate_finishing_time)))
                print("INFO: Image/sec: {}".format(config.BATCH_SIZE*config.NUM_GPUS/avg_time_per_batch))
                print("INFO: N GPUs: {}".format(config.NUM_GPUS))

    time_hist = TimeHistory()

    ws = None

    if config.warm_start_from:
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=config.warm_start_from,
            vars_to_warm_start='mesh_.*',
            var_name_to_prev_var_name={
                "mesh_decoder/dense/bias": "mesh_decoder/dense/bias",
                "mesh_decoder/dense/kernel": "mesh_decoder/dense/kernel",
                "mesh_decoder_1/mesh_conv_9/kernel": "mesh_decoder/mesh_conv_4/kernel",
                "mesh_decoder_1/mesh_conv_10/kernel": "mesh_decoder/mesh_conv_5/kernel",
                "mesh_decoder_1/mesh_conv_11/kernel": "mesh_decoder/mesh_conv_6/kernel",
                "mesh_decoder_1/mesh_conv_12/kernel": "mesh_decoder/mesh_conv_7/kernel",
                "mesh_decoder_1/mesh_conv_13/kernel": "mesh_decoder/mesh_conv_8/kernel",
                "mesh_decoder_1/mesh_re_l_u1b_8/kernel": "mesh_decoder/mesh_re_l_u1b_4/kernel",
                "mesh_decoder_1/mesh_re_l_u1b_9/kernel": "mesh_decoder/mesh_re_l_u1b_5/kernel",
                "mesh_decoder_1/mesh_re_l_u1b_10/kernel": "mesh_decoder/mesh_re_l_u1b_6/kernel",
                "mesh_decoder_1/mesh_re_l_u1b_11/kernel": "mesh_decoder/mesh_re_l_u1b_7/kernel",
                "mesh_decoder/mesh_conv/kernel": "mesh_decoder/mesh_conv_4/kernel",
                "mesh_decoder/mesh_conv_1/kernel": "mesh_decoder/mesh_conv_5/kernel",
                "mesh_decoder/mesh_conv_2/kernel": "mesh_decoder/mesh_conv_6/kernel",
                "mesh_decoder/mesh_conv_3/kernel": "mesh_decoder/mesh_conv_7/kernel",
                "mesh_decoder/mesh_conv_4/kernel": "mesh_decoder/mesh_conv_8/kernel",
                "mesh_decoder/mesh_re_l_u1b/kernel": "mesh_decoder/mesh_re_l_u1b_4/kernel",
                "mesh_decoder/mesh_re_l_u1b_1/kernel": "mesh_decoder/mesh_re_l_u1b_5/kernel",
                "mesh_decoder/mesh_re_l_u1b_2/kernel": "mesh_decoder/mesh_re_l_u1b_6/kernel",
                "mesh_decoder/mesh_re_l_u1b_3/kernel": "mesh_decoder/mesh_re_l_u1b_7/kernel",
                "mesh_encoder/dense/bias": "mesh_encoder/dense/bias",
                "mesh_encoder/dense/kernel": "mesh_encoder/dense/kernel",
                "mesh_encoder/mesh_conv_5/kernel": "mesh_encoder/mesh_conv/kernel",
                "mesh_encoder/mesh_conv_6/kernel": "mesh_encoder/mesh_conv_1/kernel",
                "mesh_encoder/mesh_conv_7/kernel": "mesh_encoder/mesh_conv_2/kernel",
                "mesh_encoder/mesh_conv_8/kernel": "mesh_encoder/mesh_conv_3/kernel",
                "mesh_encoder/mesh_re_l_u1b_4/kernel": "mesh_encoder/mesh_re_l_u1b/kernel",
                "mesh_encoder/mesh_re_l_u1b_5/kernel": "mesh_encoder/mesh_re_l_u1b_1/kernel",
                "mesh_encoder/mesh_re_l_u1b_6/kernel": "mesh_encoder/mesh_re_l_u1b_2/kernel",
                "mesh_encoder/mesh_re_l_u1b_7/kernel": "mesh_encoder/mesh_re_l_u1b_3/kernel",
            }
        )

    # Create the Estimator
    Fast_3DMM = tf.estimator.Estimator(
        model_fn=get_model_fn(config), 
        model_dir=config.LOGDIR, 
        config=config_estimator, 
        warm_start_from=ws
    )

    

    if config.test_path:
        train_spec = tf.estimator.TrainSpec(
            input_fn=get_data_fn(config, config.dataset_path, format=config.train_format), 
            max_steps=config.EPOCH_STEPS * config.TOTAL_EPOCH, 
            hooks=[time_hist]
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=get_data_fn(config, config.test_path, is_training=False, format=config.test_format),
            steps=config.EPOCH_STEPS * 5
        )

        tf.estimator.train_and_evaluate(Fast_3DMM, train_spec, eval_spec)
    else:
        Fast_3DMM.train(get_data_fn(
            config, config.dataset_path, is_training=True, format=config.train_format
        ), steps=config.EPOCH_STEPS * config.TOTAL_EPOCH, hooks=[time_hist])

    
if __name__ == '__main__':
    main()