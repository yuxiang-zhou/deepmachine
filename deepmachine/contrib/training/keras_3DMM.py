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

from menpo.shape import ColouredTriMesh, PointCloud
from menpo.image import Image
from menpo.transform import Homogeneous
from menpo3d.rasterize import rasterize_mesh
from pathlib import Path
from functools import partial
from itwmm.visualize import lambertian_shading


# deepmachine
import keras
import tensorflow as tf
import deepmachine as dm

from deepmachine.utils.machine import multi_gpu_model, enqueue_generator

    

def build_data(FLAGS, N_VERTICES, INPUT_SHAPE):
    BATCH_SIZE=FLAGS.batch_size

    features = dm.utils.union_dict([
        dm.data.provider.features.image_feature(),
        dm.data.provider.features.matrix_feature('mesh'),
        dm.data.provider.features.matrix_feature('mesh/colour'),
        dm.data.provider.features.array_feature('mesh/mask'),
    ])
    dataset = dm.data.provider.TFRecordProvider(
        FLAGS.dataset_path,
        features,
        resolvers={
            'image': dm.data.provider.resolvers.image_resolver,
            'mesh': partial(dm.data.provider.resolvers.matrix_resolver, input_shape=[N_VERTICES, 3]),
            'mesh/colour': partial(dm.data.provider.resolvers.matrix_resolver, input_shape=[N_VERTICES, 3]),
            'mesh/mask': partial(dm.data.provider.resolvers.array_resolver, input_shape=[N_VERTICES]),
        }
    )
    dataset = dm.data.provider.DatasetQueue(
        dataset, n_proccess=FLAGS.no_thread, batch_size=BATCH_SIZE)
    tf_data = dataset.get('image', 'mesh', 'mesh/colour', 'mesh/mask')

    # augmentation
    ## meshes shape: [batch_size, N_VERTICES, 3], range [-1, 1]
    def mesh_augmentation(meshes):
        # scale 
        meshes *= tf.random_uniform([BATCH_SIZE, 1, 1]) * 0.7 + 0.8

        # translation
        meshes += tf.random_uniform([BATCH_SIZE, 1, 3]) * 0.4 - 0.2
        
        # rotation
        # pending

        return meshes

    ## image shape: [256, 256, 3], range [0, 1]
    def image_augmentation(images):

        return images

    mesh_for_ae = mesh_augmentation(tf_data['mesh'])
    tf_data['image'] = image_augmentation(tf_data['image'])

    # group data
    batch_mesh_input = tf.concat([
        mesh_for_ae, tf_data['mesh/colour']
    ], axis=-1)
    batch_mesh_gt_for_ae = tf.concat([
        mesh_for_ae, tf_data['mesh/colour'], tf.reshape(tf_data['mesh/mask'], [-1,N_VERTICES,1])
    ], axis=-1)
    batch_mesh_gt_for_3dmm = tf.concat([
        tf_data['mesh'], tf_data['mesh/colour'], tf.reshape(tf_data['mesh/mask'], [-1,N_VERTICES,1])
    ], axis=-1)

    batch_image_input = tf.image.resize_images(tf_data['image'], [INPUT_SHAPE, INPUT_SHAPE])

    return [batch_mesh_input, batch_image_input], [batch_mesh_gt_for_ae, batch_mesh_gt_for_3dmm]

def build_model(FLAGS, N_VERTICES, INPUT_SHAPE, EMBEDING, FILTERS, inputs_channels=6):
    LR = FLAGS.lr
    n_gpu = len(FLAGS.gpu.split(','))
    graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices = mio.import_pickle(
        FLAGS.meta_path + '/lsfm_LDUA.pkl', encoding='latin1')

    # define components
    ## image encoder
    def build_img_encoder():
        input_img = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_img')

        img_embedding = dm.networks.Encoder2D(
            input_img, EMBEDING, depth=4, nf=32)

        return dm.Model(input_img, img_embedding, name='image_encoder')

    ## mesh encoder
    def build_mesh_encoder():
        input_mesh = dm.layers.Input(shape=[N_VERTICES, inputs_channels], name='input_mesh')
        mesh_embedding = dm.networks.MeshEncoder(
            input_mesh, EMBEDING, graph_laplacians, downsampling_matrices, filter_list=FILTERS)

        return dm.Model(input_mesh, mesh_embedding, name='mesh_encoder')

    ## common decoder
    def build_decoder():
        input_embeding = dm.layers.Input(shape=[EMBEDING], name='input_embeding')
        output_mesh = dm.networks.MeshDecoder(
            input_embeding, 
            inputs_channels, 
            graph_laplacians, 
            adj_matrices, 
            upsamling_matrices, 
            polynomial_order=6, 
            filter_list=FILTERS)

        return dm.Model(input_embeding, output_mesh, name='decoder')

    # Mesh AE stream
    ## define inputs
    input_mesh = dm.layers.Input(shape=[N_VERTICES, 6], name='input_mesh')
    input_image = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_image')

    ## define components
    img_encoder_model = build_img_encoder()
    mesh_encoder_model = build_mesh_encoder()
    decoder_model = build_decoder()

    ## define connections
    output_mesh_ae = decoder_model(mesh_encoder_model(input_mesh))
    output_mesh_3dmm = decoder_model(img_encoder_model(input_image))

    ## custom losses
    def masked_mae(gt_y, pred_y):
        gt_mask = gt_y[...,-1:]
        gt_mesh = gt_y[...,:-1]
        return dm.losses.mae(gt_mesh*gt_mask, pred_y*gt_mask)


    # model definition
    model_ae_mesh = dm.DeepMachine(
        inputs=[input_mesh], 
        outputs=[output_mesh_ae],
        name='MeshAutoEncoder'
    )
    if n_gpu > 1:
        model_ae_mesh = multi_gpu_model(model_ae_mesh, gpus=n_gpu)

    model_ae_mesh.compile(
        optimizer=dm.optimizers.Adam(lr=LR),
        loss=[masked_mae]
    )

    mesh_encoder_model.trainable = False
    decoder_model.trainable = False
    model_ae_mesh.trainable = False
    model_3dmm = dm.DeepMachine(
        inputs=[input_image], 
        outputs=[output_mesh_3dmm],
        name='3DMM'
    )

    ## multi gpu support
    if n_gpu > 1:
        model_3dmm = multi_gpu_model(model_3dmm, gpus=n_gpu)

    ## compile mesh stream
    model_3dmm.compile(
        optimizer=dm.optimizers.Adam(lr=LR*FLAGS.m_weight),
        loss=[masked_mae]
    )

    return model_ae_mesh, model_3dmm, decoder_model

def main():
    # flag definitions
    tf.app.flags.DEFINE_string('meta_path', '/vol/atlas/homes/yz4009/databases/mesh/meta', '''path to meta files''')
    tf.app.flags.DEFINE_integer('embedding', 128, '''embedding''')
    tf.app.flags.DEFINE_float('m_weight', 1.0, '''embedding''')
    tf.app.flags.DEFINE_string('pretrained_meshae', '/homes/yz4009/db/trained_model/mesh_ae/weights.h5', '''pretrained_meshae''')
    from deepmachine.flags import FLAGS

    def format_folder(FLAGS):
        post_fix = 'lr{:02.1f}_d{:02.1f}_emb{:03d}_b{:02d}_3dmmw{:.3f}'.format(
            FLAGS.lr*1000, FLAGS.lr_decay * 100, FLAGS.embedding, FLAGS.batch_size, FLAGS.m_weight
        )

        logdir = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
            FLAGS.logdir, post_fix
        )

        return logdir

    # hyperparameters
    BATCH_SIZE = FLAGS.batch_size
    LR = FLAGS.lr
    EMBEDING = FLAGS.embedding
    LOGDIR = format_folder(FLAGS)
    N_VERTICES = 53215
    INPUT_SHAPE = 112
    STARTING_3DMM = -1 if FLAGS.pretrained_meshae else 100
    FILTERS = [16, 32, 32, 64]

    # global variables
    shape_model = mio.import_pickle(
        FLAGS.meta_path + '/all_all_all.pkl')
    trilist = shape_model.instance([]).trilist

    # ### Training
    def train_op(models, data, i_epoch, i_batch, epoch_end, training_history=None, **kwargs):
        model_ae_mesh, model_3dmm, _ = models
        [input_mesh, input_image], [mesh_gt, _] = dm.engine.training.tf_dataset_adapter(data)
        sess = dm.K.get_session()
        # ----------------------
        #  Train AutoEncoder
        # ----------------------
        loss_ae = model_ae_mesh.train_on_batch(input_mesh, mesh_gt)

        # ------------------
        #  Train 3DMM
        # ------------------
        if i_epoch > STARTING_3DMM:
            loss_3dmm = model_3dmm.train_on_batch(input_image, mesh_gt)
        else:
            loss_3dmm = 0

        logs = dm.utils.Summary(
            {
                "losses/AE": loss_ae,
                "losses/3dmm": loss_3dmm,
                "learning_rate/ae": model_ae_mesh.optimizer.lr.eval(sess),
                "learning_rate/3dmm": model_3dmm.optimizer.lr.eval(sess),
            }
        )

        if epoch_end:
            t_rot30l = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(30), [1,0,0]))
            t_rot30r = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(-30), [1,0,0]))

            ae_mesh = model_ae_mesh.predict(input_mesh)
            pred_mesh = model_3dmm.predict(input_image)
            logs.update_images({
                'inputs/image': input_image,
                'pred/mesh/AE': dm.utils.mesh.render_meshes(ae_mesh[:4], trilist),
                'pred/mesh/3dmm/r0': dm.utils.mesh.render_meshes(pred_mesh[:4], trilist),
                'pred/mesh/3dmm/r+30': dm.utils.mesh.render_meshes(
                    [np.concatenate([t_rot30l.apply(m[:,:3]), m[:,3:]], axis=-1) for m in pred_mesh[:4]], trilist),
                'pred/mesh/3dmm/r-30': dm.utils.mesh.render_meshes(
                    [np.concatenate([t_rot30r.apply(m[:,:3]), m[:,3:]], axis=-1) for m in pred_mesh[:4]], trilist),
                'gt/mesh': dm.utils.mesh.render_meshes(input_mesh[:4], trilist),
            })

        return logs


    train_tfrecord = build_data(FLAGS, N_VERTICES, INPUT_SHAPE)
    models = build_model(FLAGS, N_VERTICES, INPUT_SHAPE, EMBEDING, FILTERS)

    if FLAGS.pretrained_meshae:
        models[0].load_weights(FLAGS.pretrained_meshae)

    lr_decay_m1 = dm.callbacks.LearningRateScheduler(
        schedule=lambda epoch: LR * FLAGS.lr_decay ** epoch)
    lr_decay_m1.set_model(models[0])
    lr_decay_m2 = dm.callbacks.LearningRateScheduler(
        schedule=lambda epoch: LR * FLAGS.lr_decay ** epoch)
    lr_decay_m2.set_model(models[1])

    history = dm.engine.training.train_monitor(
        models,
        train_tfrecord, 
        train_op,
        epochs=200, 
        step_per_epoch=4995,
        callbacks=[
            lr_decay_m1,
            lr_decay_m2
        ],
        verbose=FLAGS.verbose,
        logdir=LOGDIR,
    )

if __name__ == '__main__':
    main()
