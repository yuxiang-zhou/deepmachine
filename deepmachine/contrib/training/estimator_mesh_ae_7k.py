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

def get_data_fn(config, dataset_path, is_training=True, data_format='h5py', num_epochs=None):
    BATCH_SIZE = config.BATCH_SIZE
    NUM_GPUS = config.NUM_GPUS

    def data_fn_h5py():
        
        with h5py.File(dataset_path) as df:
            if is_training:
                all_cmesh = df['colour_mesh'][:8000]
            else:
                all_cmesh = df['colour_mesh'][8000:]
            
            L,D,U,A = mio.import_pickle('/vol/phoebe/yz4009/databases/mesh/meta/mein3dcrop_LDUA.pkl', encoding='latin1')
            all_cmesh = D[0].dot(all_cmesh.transpose(
                [1,0,2]).reshape([D[0].shape[1],-1])).reshape([D[0].shape[0], -1, config.inputs_channels]).transpose([1,0,2])
            all_cmesh = all_cmesh.astype(np.float32)
            
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                "cmesh": all_cmesh,
            }, y={
                "mask": np.ones_like(all_cmesh)[...,:1]
            }, shuffle=True, batch_size=BATCH_SIZE, num_epochs=num_epochs, num_threads=config.no_thread
        )

        return input_fn

    return data_fn_h5py()


def get_model_fn(config):
    BATCH_SIZE = config.BATCH_SIZE
    def model_fn(features, labels, mode, params):
        # define components

        # mesh encoder
        input_mesh = features['cmesh']
        if not config.use_colour:
            input_mesh = input_mesh[..., :3]


        mesh_embedding = dm.networks.tfl.MeshEncoder(
            input_mesh, 
            config.EMBEDING, 
            config.graph_laplacians, 
            config.downsampling_matrices, 
            polynomial_order=config.NK,
            filter_list=config.FILTERS)

        # decoder
        output_rec_mesh = dm.networks.tfl.MeshDecoder(
            mesh_embedding,
            config.inputs_channels,
            config.graph_laplacians,
            config.adj_matrices,
            config.upsamling_matrices,
            polynomial_order=config.NK,
            filter_list=config.FILTERS
        )

        # PREDICT mode
        # Build estimator spec
        predictions = {
            'cmesh': output_rec_mesh
        }
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


        # Build Additional Graph and Loss (for both TRAIN and EVAL modes)
        # camera position:
        eye = tf.constant(BATCH_SIZE * [[0.0, 0.0, -2.0]], dtype=tf.float32)
        center = tf.constant(BATCH_SIZE * [[0.0, 0.0, 0.0]], dtype=tf.float32)
        world_up = tf.constant(BATCH_SIZE * [[-1.0, 0.0, 0.0]], dtype=tf.float32)
        ambient_colors = tf.constant(BATCH_SIZE * [[1., 1., 1.]], dtype=tf.float32) * 0.1
        light_positions = tf.constant(BATCH_SIZE * [[[2.0, 0, -2.0], [-2.0, 0, -2.0]]])
        light_intensities = tf.ones([BATCH_SIZE, 1, 3], dtype=tf.float32)
        mesh_triangles = tf.constant(config.trilist, dtype=tf.int32)

        ## predicted mesh render
        pred_mesh_v = output_rec_mesh[...,:3]
        if config.use_colour:
            pred_mesh_c = output_rec_mesh[...,3:]
        else:
            pred_mesh_c = tf.ones_like(pred_mesh_v) * 0.5
        pred_mesh_c = tf.clip_by_value(pred_mesh_c,0,1)
        pred_mesh_v.set_shape([BATCH_SIZE, config.N_VERTICES, 3])
        pred_mesh_n = tf.nn.l2_normalize(pred_mesh_v, axis=2)
        pred_mesh_n.set_shape([BATCH_SIZE, config.N_VERTICES, 3])

        rendered_mesh_rec = mesh_renderer(
            pred_mesh_v,
            triangles=mesh_triangles,
            normals=pred_mesh_n,
            diffuse_colors=pred_mesh_c,
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

        tf.summary.image('image/mesh/pred', rendered_mesh_rec)

        ## predicted mesh render
        gt_mesh_v = input_mesh[...,:3]
        if config.use_colour:
            gt_mesh_c = input_mesh[...,3:]
        else:
            gt_mesh_c = tf.ones_like(gt_mesh_v) * 0.5

        gt_mesh_v.set_shape([BATCH_SIZE, config.N_VERTICES, 3])
        gt_mesh_n = tf.nn.l2_normalize(gt_mesh_v, axis=2)
        gt_mesh_n.set_shape([BATCH_SIZE, config.N_VERTICES, 3])
        

        rendered_mesh_gt = mesh_renderer(
            gt_mesh_v,
            triangles=mesh_triangles,
            normals=gt_mesh_n,
            diffuse_colors=gt_mesh_c,
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

        tf.summary.image('image/mesh/gt', rendered_mesh_gt)

        # define losses
        if config.use_colour:
            loss_shape = tf.reduce_mean(tf.losses.absolute_difference(
                features['cmesh'][...,:3], output_rec_mesh[...,:3]
            ))

            loss_appearance = tf.reduce_mean(tf.losses.absolute_difference(
                features['cmesh'][...,3:], output_rec_mesh[...,3:], weights=labels['mask']
            ))
           
            tf.summary.scalar('loss/shape', loss_shape)
            tf.summary.scalar('loss/appearance', loss_appearance)

            total_loss = loss_shape + 0.5*loss_appearance
            total_loss /= 2.
        else:
            total_loss = tf.reduce_mean(tf.losses.absolute_difference(
                features['cmesh'][...,:3], output_rec_mesh, weights=labels['mask']
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

            if config.opt == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif config.opt == 'sgd':
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
            elif config.opt == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            else:
                raise Exception('Undefined optimizer: ' + config.opt)

            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
            train_op = optimizer.minimize(
                loss=total_loss,
                global_step=global_steps,
            )

            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        
        def nmse(labels, predictions, norm_factor=0.61775377381):
            error, op = tf.metrics.mean_squared_error(labels=labels, predictions=predictions)
            return error / norm_factor, op
            

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "Reconstruction_ALL_mae": tf.metrics.mean_absolute_error(labels=features['cmesh'], predictions=predictions["cmesh"]),
            "Reconstruction_MESH_mae": tf.metrics.mean_absolute_error(labels=features['cmesh'][...,:3], predictions=predictions["cmesh"][...,:3]),
            "Reconstruction_MESH_mse": tf.metrics.mean_squared_error(labels=features['cmesh'][...,:3], predictions=predictions["cmesh"][...,:3]), 
            "Reconstruction_MESH_nmse": nmse(labels=features['cmesh'][...,:3], predictions=predictions["cmesh"][...,:3]),
            "Reconstruction_COLOUR_mae": tf.metrics.mean_absolute_error(labels=features['cmesh'][...,3:], predictions=predictions["cmesh"][...,3:]),
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)

    return model_fn


def get_config():

    # flag definitions
    tf.app.flags.DEFINE_string(
        'meta_path', '/vol/phoebe/yz4009/databases/mesh/meta', '''path to meta files''')
    tf.app.flags.DEFINE_string(
        'test_path', '', '''path to test files''')
    tf.app.flags.DEFINE_string(
        'warm_start_from', None, '''path to test files''')
    tf.app.flags.DEFINE_string(
        'opt', 'adam', '''path to test files''')
    tf.app.flags.DEFINE_boolean(
        'colour', True, '''Whether to use colours''')
    tf.app.flags.DEFINE_integer('embedding', 128, '''embedding''')
    tf.app.flags.DEFINE_integer('k_poly', 4, '''embedding''')
    from deepmachine.flags import FLAGS

    class Config:

        def format_folder(self, FLAGS):
            post_fix = 'lr_{:f}-d_{:f}-emb_{:03d}-b_{:02d}-{}-k_{}-colour_{}'.format(
                FLAGS.lr, FLAGS.lr_decay, FLAGS.embedding, FLAGS.batch_size, FLAGS.opt, FLAGS.k_poly, str(FLAGS.colour)
            )

            logdir = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
                FLAGS.logdir, post_fix
            )

            return logdir

        def __init__(self, *args, **kwargs):
            # hyperparameters
            self.BATCH_SIZE = FLAGS.batch_size
            self.LR = FLAGS.lr
            self.opt = FLAGS.opt
            self.NK = FLAGS.k_poly
            self.use_colour = FLAGS.colour
            self.inputs_channels = 6 if FLAGS.colour else 3
            self.lr_decay = FLAGS.lr_decay
            self.warm_start_from = FLAGS.warm_start_from
            self.EMBEDING = FLAGS.embedding
            self.LOGDIR = self.format_folder(FLAGS)
            self.NUM_GPUS = len(FLAGS.gpu.split(','))
            self.TOTAL_EPOCH = FLAGS.n_epoch
            self.no_thread = FLAGS.no_thread
            self.dataset_path = FLAGS.dataset_path
            self.test_path = FLAGS.test_path
            self.train_format = 'tf' if FLAGS.dataset_path.endswith('.tfrecord') else 'h5py'
            self.test_format = 'tf' if FLAGS.test_path.endswith('.tfrecord') else 'h5py'
            # constant
            self.INPUT_SHAPE = 112
            self.FILTERS = [16, 16, 16, 32]
            self.DB_SIZE = 8000
            self.EPOCH_STEPS = self.DB_SIZE // (self.BATCH_SIZE * self.NUM_GPUS)
            
            # graph constant
            self.luda_file = 'mein3dcrop_7k_LDUA.pkl'
            # self.luda_file = 'lsfm_LDUA.pkl'
            self.symetric_index = np.load(FLAGS.meta_path + '/symetric_index_full.npy').astype(np.int32)
            self.shape_model = mio.import_pickle(FLAGS.meta_path + '/mein3dcrop_7k_mesh.pkl', encoding='latin1')
            self.trilist = self.shape_model.trilist
            self.graph_laplacians, self.downsampling_matrices, self.upsamling_matrices, self.adj_matrices = mio.import_pickle(
                FLAGS.meta_path + '/' + self.luda_file, encoding='latin1')
            self.N_VERTICES = self.graph_laplacians[0].shape[0]


    return Config()

def main():

    config = get_config()
    # configuration
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=config.NUM_GPUS) if config.NUM_GPUS > 1 else None
    config_estimator = tf.estimator.RunConfig(
        train_distribute=strategy,
        save_checkpoints_steps=config.EPOCH_STEPS,
        save_summary_steps=100,
        keep_checkpoint_max=5,
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

    # Create the Estimator
    Fast_3DMM = tf.estimator.Estimator(
        model_fn=get_model_fn(config), model_dir=config.LOGDIR, config=config_estimator, warm_start_from=config.warm_start_from)

    
    if config.test_path:
        train_spec = tf.estimator.TrainSpec(
            input_fn=get_data_fn(config, config.dataset_path, is_training=True, data_format=config.train_format), 
            max_steps=config.EPOCH_STEPS * config.TOTAL_EPOCH, 
            hooks=[time_hist]
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=get_data_fn(config, config.test_path, is_training=False, data_format=config.test_format)
        )

        tf.estimator.train_and_evaluate(Fast_3DMM, train_spec, eval_spec)
    else:
        Fast_3DMM.train(get_data_fn(
            config, config.dataset_path, is_training=True, data_format=config.train_format
        ), steps=config.EPOCH_STEPS * config.TOTAL_EPOCH, hooks=[time_hist])

    



if __name__ == '__main__':
    main()