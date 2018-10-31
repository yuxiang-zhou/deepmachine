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

# flag definitions
tf.app.flags.DEFINE_string('meta_path', '/vol/atlas/homes/yz4009/databases/mesh/meta', '''path to meta files''')
tf.app.flags.DEFINE_integer('embedding', 128, '''embedding''')
from deepmachine.flags import FLAGS
    

def main():

    def format_folder(FLAGS):
        post_fix = 'lr{:02.1f}_d{:02.1f}_emb{:03d}_b{:02d}'.format(
            FLAGS.lr*1000, FLAGS.lr_decay * 100, FLAGS.embedding, FLAGS.batch_size
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
    FILTERS = [16, 32, 32, 64]
    

    # globel constant
    n_gpu = len(FLAGS.gpu.split(','))
    shape_model = mio.import_pickle(
        FLAGS.meta_path + '/all_all_all.pkl')
    trilist = shape_model.instance([]).trilist
    graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices = mio.import_pickle(
        FLAGS.meta_path + '/lsfm_LDUA.pkl', encoding='latin1')

    def build_data():
        
        class H5Data(dm.utils.Sequence):

            def __init__(self, fp, batch_size=BATCH_SIZE):
                self.train_data = h5py.File(fp, 'r')
                self.batch_size = batch_size
                self.size = self.train_data['image'].len()
                self.indexes = list(range(self.size))
                np.random.shuffle(self.indexes)
                super().__init__()

            def __len__(self):
                return self.size // self.batch_size

            def __getitem__(self, idx):
                indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

                batch_mesh = []
                batch_mesh_colour = []
                batch_mesh_mask = []
                batch_img = []
                for i in indexes:
                    if self.train_data['label'][i] == 1:
                        try:
                            # mesh
                            batch_mesh.append(
                                self.train_data['mesh'][i]
                            )

                            batch_mesh_colour.append(
                                self.train_data['mesh_colour'][i]
                            )

                            batch_mesh_mask.append(
                                self.train_data['mesh_mask'][i].reshape([-1,1])
                            )

                            # images
                            batch_img.append(
                                self.train_data['image'][i]
                            )
                        except:
                            pass

                batch_mesh = np.array(batch_mesh)
                batch_mesh_colour = np.array(batch_mesh_colour)
                batch_mesh_mask = np.array(batch_mesh_mask)
                batch_img = np.array(batch_img)

                batch_mesh_input = np.concatenate([
                    batch_mesh, batch_mesh_colour
                ], axis=-1)

                batch_mesh_gt = np.concatenate([
                    batch_mesh, batch_mesh_colour, batch_mesh_mask.astype(np.float)
                ], axis=-1)

                return [batch_mesh_input, batch_img], [batch_mesh_gt, batch_mesh_gt]

            def on_epoch_end(self, *args, **kwargs):
                np.random.shuffle(self.indexes)
                return super().on_epoch_end()
        
        return H5Data(FLAGS.dataset_path, batch_size=BATCH_SIZE)

    def build_model(inputs_channels=6, n_gpu=n_gpu):

        # define components
        ## image encoder
        def build_img_encoder():
            input_img = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_img')

            img_embedding = dm.networks.Encoder2D(
                input_img, EMBEDING, depth=4, nf=64)

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

        model_3dmm = dm.DeepMachine(
            inputs=[input_mesh, input_image], 
            outputs=[output_mesh_ae, output_mesh_3dmm],
            name='MeshStream'
        )

        ## multi gpu support
        if n_gpu > 1:
            model_3dmm = multi_gpu_model(model_3dmm, gpus=n_gpu)

        ## compile mesh stream
        def masked_mae(gt_y, pred_y):
            gt_mask = gt_y[...,-1:]
            gt_mesh = gt_y[...,:-1]
            return dm.losses.mae(gt_mesh*gt_mask, pred_y*gt_mask)

        model_3dmm.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=[masked_mae, masked_mae]
        )

        return model_3dmm


    def custom_summary(train_in, train_out, predict_y):

        return {
            'input/mesh': dm.utils.mesh.render_meshes(train_in[0][:4], trilist),
            'pred/mesh': dm.utils.mesh.render_meshes(predict_y[0][:4], trilist),
            'gt/mesh': dm.utils.mesh.render_meshes(train_out[0][:4,:,:6], trilist),
         }

    training_generator = build_data()
    model_3dmm = build_model()

    results = model_3dmm.fit(
        training_generator,
        epochs=400,
        lr_decay=FLAGS.lr_decay,
        logdir=LOGDIR,
        verbose=2,
        workers=FLAGS.no_thread,
        summary_ops=[custom_summary]
    )

if __name__ == '__main__':
    main()
