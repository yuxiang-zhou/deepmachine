# basic library
import os
import shutil
import math
import time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
import h5py

from menpo.shape import ColouredTriMesh
from menpo.transform import Homogeneous
from menpo3d.rasterize import rasterize_mesh
from pathlib import Path
from functools import partial
from itwmm.visualize import lambertian_shading


# deepmachine
import keras
import tensorflow as tf
import deepmachine as dm

# flag definitions
tf.app.flags.DEFINE_string('ref_model', 'coma', '''One of "coma, lsfm, 4dfab"''')
from deepmachine.flags import FLAGS


def main():
    # hyperparameters
    BATCH_SIZE = FLAGS.batch_size
    N_VERTICES = 5023
    EMBEDING = 8
    LR = FLAGS.lr
    FILTERS = [16, 16, 16, 32]
    LOGDIR = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
        FLAGS.logdir, int(time.time()))


    # referece model
    if FLAGS.ref_model.startswith('lsfm') or FLAGS.ref_model == 'mein3d':
        N_VERTICES = 53215
        shape_model = mio.import_pickle(
            '/homes/yz4009/wd/notebooks/Projects/MLProjects/models/all_all_all.pkl')
        trilist = shape_model.instance([]).trilist
        EMBEDING = 128
        FILTERS = [16, 32, 32, 64]

    elif FLAGS.ref_model == 'mein3dcrop':
        N_VERTICES = 28431
        face_mean_crop = m3io.import_mesh('/homes/yz4009/wd/gitdev/coma/data/face_mean_mesh_crop.obj')
        trilist = face_mean_crop.trilist
        EMBEDING = 128
        FILTERS = [16, 32, 32, 64]
        
    elif FLAGS.ref_model == 'coma':
        N_VERTICES = 5023
        trilist = mio.import_pickle(
            '/homes/yz4009/wd/gitdev/coma/data/coma_f.pkl', encoding='latin1')

    elif FLAGS.ref_model == '4dfab':
        N_VERTICES = 2064
        template_mesh = m3io.import_mesh('/vol/atlas/homes/Shiyang/CVPR18/code/animation/COMA/coma/data/afm_l1_cropped_final.obj')
        trilist = template_mesh.trilist

    else:
        raise Exception('Undefined ref_model: {}'.format(FLAGS.ref_model))

    graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices = mio.import_pickle(
        '/homes/yz4009/wd/gitdev/coma/data/{}_LDUA.pkl'.format(FLAGS.ref_model), encoding='latin1')

    def build_data():
        class MeshRandomSample(dm.utils.Sequence):

            def __init__(self, batch_size=BATCH_SIZE):
                self.batch_size = batch_size
                self.size = 10000 // BATCH_SIZE
                super().__init__()

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                batch_sample_mesh = np.array([
                    shape_model.instance(np.random.sample([50]) * 3, normalized_weights=True).points for _ in range(self.batch_size)
                ])

                return [batch_sample_mesh], [batch_sample_mesh]

        class NumpyMesh(dm.utils.Sequence):

            def __init__(self, fp, batch_size=BATCH_SIZE, scale=1, normalize=False):
                self.train_mesh = np.load(fp)
                if normalize:
                    self.train_mesh /= np.max(np.abs([self.train_mesh.max(), self.train_mesh.min()])) + 1
                self.batch_size = batch_size
                self.size = self.train_mesh.shape[0]
                self.indexes = list(range(self.size))
                self.scale = scale
                np.random.shuffle(self.indexes)
                super().__init__()

            def __len__(self):
                return self.size // self.batch_size

            def __getitem__(self, idx):
                batch_sample_mesh = np.array([
                    self.train_mesh[self.indexes[i]] * self.scale for i in range(idx, idx+BATCH_SIZE)
                ])

                return [batch_sample_mesh], [batch_sample_mesh]

            def on_epoch_end(self, *args, **kwargs):
                np.random.shuffle(self.indexes)
                return super().on_epoch_end()

        class H5Mesh(dm.utils.Sequence):

            def __init__(self, fp, dataset, batch_size=BATCH_SIZE, with_rgb=True):
                self.train_mesh = h5py.File(fp, 'r')[dataset]
                self.batch_size = batch_size
                self.size = self.train_mesh.len()
                self.indexes = list(range(self.size))
                self.with_rgb = with_rgb
                np.random.shuffle(self.indexes)
                super().__init__()

            def __len__(self):
                return self.size // self.batch_size

            def __getitem__(self, idx):
                batch_sample_mesh = np.array([
                    self.train_mesh[self.indexes[i]] for i in range(idx, idx+BATCH_SIZE)
                ])

                if not self.with_rgb:
                    batch_sample_mesh = batch_sample_mesh[...,:3]

                return [batch_sample_mesh], [batch_sample_mesh]

            def on_epoch_end(self, *args, **kwargs):
                np.random.shuffle(self.indexes)
                return super().on_epoch_end()

        if FLAGS.ref_model == 'lsfm':
            return MeshRandomSample(batch_size=BATCH_SIZE)

        elif FLAGS.ref_model == 'lsfm_rgb':
            return H5Mesh('/homes/yz4009/wd/gitdev/coma/data/lsfm_texture_train.h5', 'lsfm_colour', batch_size=BATCH_SIZE)

        elif FLAGS.ref_model == 'mein3d':
            return H5Mesh('/homes/yz4009/wd/gitdev/coma/data/mein3d.h5', 'colour_mesh', batch_size=BATCH_SIZE)

        elif FLAGS.ref_model == 'mein3dcrop':
            return H5Mesh('/homes/yz4009/wd/gitdev/coma/data/mein3dcrop.h5', 'colour_mesh', batch_size=BATCH_SIZE)
            
        elif FLAGS.ref_model == 'coma':
            return NumpyMesh('/homes/yz4009/wd/gitdev/coma/data/bareteeth/train.npy', batch_size=BATCH_SIZE, scale=5)

        elif FLAGS.ref_model == '4dfab':
            return NumpyMesh('/vol/atlas/homes/Shiyang/CVPR18/code/animation/COMA/coma/data/RECON4DFAB/AFM/train.npy', batch_size=BATCH_SIZE, normalize=True)

        return None

    def build_model(inputs_channels=3):
        input_mesh = dm.layers.Input(shape=[N_VERTICES, inputs_channels], name='input_mesh')

        mesh_embedding = dm.networks.MeshEncoder(
            input_mesh, EMBEDING, graph_laplacians, downsampling_matrices, filter_list=FILTERS)
        output_mesh = dm.networks.MeshDecoder(
            mesh_embedding, 
            inputs_channels, 
            graph_laplacians, 
            adj_matrices, 
            upsamling_matrices, 
            polynomial_order=6, 
            filter_list=FILTERS)
        
        # wrapping input and output
        mesh_ae = dm.DeepMachine(
            inputs=input_mesh, 
            outputs=[output_mesh]
        )
        
        # compile model with optimizer
        mesh_ae.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=['mae']
        )
        
        # ---------------- rendering layer ------------
        mesh_to_render = dm.layers.Input(shape=[N_VERTICES, 3], name='mesh_to_render')
        mesh_to_render.set_shape([BATCH_SIZE, N_VERTICES, 3])
        vertex_color = dm.layers.Input(shape=[N_VERTICES, 3], name='vertex_color')
        vertex_color.set_shape([BATCH_SIZE, N_VERTICES, 3])

        # Build vertices and normals
        mesh_vertices = mesh_to_render
        mesh_vertices.set_shape([BATCH_SIZE, N_VERTICES, 3])
        mesh_normals = tf.nn.l2_normalize(mesh_vertices, axis=2)
        mesh_normals.set_shape([BATCH_SIZE, N_VERTICES, 3])

        # random model transformation
        model_transforms = dm.utils.camera_utils.euler_matrices(
            tf.random_uniform([BATCH_SIZE, 3]) * np.pi / 2 - np.pi / 4.
        )[:, :3, :3]

        # rendering output
        mesh_triangles = tf.constant(trilist, dtype=tf.int32)
        
        # camera position:
        eye = tf.constant(BATCH_SIZE * [[0.0, 0.0, -2.0]], dtype=tf.float32)
        center = tf.constant(BATCH_SIZE * [[0.0, 0.0, 0.0]], dtype=tf.float32)
        world_up = tf.constant(BATCH_SIZE * [[1.0, 0.0, 0.0]], dtype=tf.float32)
        ambient_colors = tf.constant(BATCH_SIZE * [[1., 1., 1.]], dtype=tf.float32) * 0.1
        light_positions = tf.constant(BATCH_SIZE * [[[2.0, 2.0, 2.0]]]) * 3.
        light_intensities = tf.ones([BATCH_SIZE, 1, 3], dtype=tf.float32)

        render_mesh = dm.layers.Renderer(
            # image size
            image_width=256,
            image_height=256,
            # mesh definition
            triangles=mesh_triangles,
            normals=mesh_normals,
            # colour definition
            diffuse_colors=vertex_color,
            ambient_color=ambient_colors,
            # camera definition
            camera_position=eye,
            camera_lookat=center,
            camera_up=world_up,
            # light definition
            light_positions=light_positions,
            light_intensities=light_intensities,
        )(mesh_vertices)

        mesh_render = dm.DeepMachine(
            inputs=[mesh_to_render, vertex_color], 
            outputs=[render_mesh]
        )
        # ----------------------

        
        return mesh_ae, mesh_render

    def custom_summary(train_x, train_y, predict_y):

        def render_mesh(sample_mesh, res=256, scale=1):

            if sample_mesh.shape[-1] != 3:
                sample_colours = sample_mesh[...,3:]
            else:
                sample_colours = np.ones_like(sample_mesh) * [0, 0, 1]
            sample_mesh = sample_mesh[...,:3]

            sample_mesh = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(90),[0,0,-1])).apply(sample_mesh)
            sample_mesh = ColouredTriMesh(
                sample_mesh * scale * res / 2 + res / 2,
                trilist=trilist,
                colours=sample_colours
            )
            sample_mesh = lambertian_shading(sample_mesh, ambient_colour=0)
            m3io.export_mesh(sample_mesh, Path(LOGDIR)/'{}.obj'.format(time.time()))

            mesh_img = rasterize_mesh(
                sample_mesh,
                [res, res]
            )
            mesh_img = mesh_img.rotate_ccw_about_centre(90)

            return mesh_img.pixels_with_channels_at_back()

        def dr_render_mesh(sample_mesh):
            if sample_mesh.shape[-1] != 3:
                sample_colours = sample_mesh[...,3:]
            else:
                sample_colours = np.ones_like(sample_mesh) * [0, 0, 1]
            sample_mesh = sample_mesh[...,:3]

            return renderer.predict([sample_mesh, sample_colours])[...,:3]


        return {
            'target/mesh': np.array(list(map(render_mesh, train_y[0]))),
            'output/mesh': np.array(list(map(render_mesh, predict_y[0]))),
            'target/dr_mesh': dr_render_mesh(train_y[0]),
            'output/dr_mesh': dr_render_mesh(predict_y[0]),
        }

    training_generator = build_data()
    auto_encoder, renderer = build_model(inputs_channels=6 if FLAGS.ref_model == 'lsfm_rgb' or FLAGS.ref_model.startswith('mein3d') else 3)

    results = auto_encoder.fit(
        training_generator,
        epochs=300,
        lr_decay=FLAGS.lr_decay,
        logdir=LOGDIR,
        verbose=2,
        workers=FLAGS.no_thread,
        summary_ops=[custom_summary]
    )


if __name__ == '__main__':
    main()
