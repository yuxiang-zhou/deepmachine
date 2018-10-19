# basic library
import os
import shutil
import math
import time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
from menpo.shape import ColouredTriMesh
from menpo.image import Image
from menpo.transform import Homogeneous, Scale, Translation
from menpo3d.rasterize import rasterize_mesh
from menpo3d.unwrap import optimal_cylindrical_unwrap
from pathlib import Path
from functools import partial
from itwmm.visualize import lambertian_shading


# deepmachine
import keras
import tensorflow as tf
import deepmachine as dm

# flag definitions
from deepmachine.flags import FLAGS


def main():
    # hyperparameters
    BATCH_SIZE = FLAGS.batch_size
    N_VERTICES = 5023
    INPUT_SHAPE = 256
    EMBEDING = 8
    MESH_SCALE = 1
    LR = FLAGS.lr
    LOGDIR = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
        FLAGS.logdir, int(time.time()))

    # global variables
    shape_model = mio.import_pickle(
        '/homes/yz4009/wd/notebooks/Projects/MLProjects/models/all_all_all.pkl')
    mean_mesh = shape_model.instance([])
    trilist = mean_mesh.trilist
    graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices = mio.import_pickle(
        '/homes/yz4009/wd/gitdev/coma/data/lsfm_LDUA.pkl', encoding='latin1')

    def build_data():
        # ### load models
        uv_shape = (INPUT_SHAPE, INPUT_SHAPE)
        UV_template = optimal_cylindrical_unwrap(mean_mesh).apply(mean_mesh)
        UV_template.points = UV_template.points[:, [1, 0]]
        UV_template.points[:, 0] = UV_template.points[:, 0].max() - UV_template.points[:, 0]

        UV_template.points -= UV_template.points.min(axis=0)
        UV_template.points /= UV_template.points.max(axis=0)
        UV_template.points *= np.array([uv_shape])

        class MeshRandomSample(dm.utils.Sequence):

            def random_sample_UV(self, n_pcs = 50, random_angle = 45, shape = [INPUT_SHAPE, INPUT_SHAPE]):
                # random mesh
                mesh_instance = shape_model.instance(np.random.sample([n_pcs]) * 3, normalized_weights=True)
                sample_mesh = ColouredTriMesh(mesh_instance.points, trilist=mesh_instance.trilist)
                sample_mesh = lambertian_shading(sample_mesh)
                sample_mesh = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(90), [0,0,1])).apply(sample_mesh)
                sample_mesh = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(180), [1,0,0])).apply(sample_mesh)
                sample_mesh = Scale([100,100,100]).apply(sample_mesh)

                # random rotation and scale and translation
                sample_mesh = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(np.random.rand() * random_angle - random_angle / 2), [0,0,1])).apply(sample_mesh)
                sample_mesh = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(np.random.rand() * random_angle - random_angle / 2), [0,1,0])).apply(sample_mesh)
                sample_mesh = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(np.random.rand() * random_angle - random_angle / 2), [1,0,0])).apply(sample_mesh)
                sample_mesh = Scale(np.tile(np.random.sample([1]), [3]) * 1. + 0.5).apply(sample_mesh)

                # translate to image centre
                sample_mesh = Translation([shape[0] / 2,shape[1] / 2,0]).apply(sample_mesh)
                
                # random perturbation
                sample_mesh = Translation(np.random.sample([3]) * 128 - 64).apply(sample_mesh)

                u_mesh = ColouredTriMesh(
                    sample_mesh.points, 
                    trilist=sample_mesh.trilist, 
                    colours=UV_template.points[:,0,None])
                u_image = rasterize_mesh(u_mesh, shape)
                v_mesh = ColouredTriMesh(
                    sample_mesh.points, 
                    trilist=sample_mesh.trilist, 
                    colours=UV_template.points[:,1,None])
                v_image = rasterize_mesh(v_mesh, shape)

                IUV_image = Image(np.concatenate([u_image.mask.pixels,u_image.pixels / 255., v_image.pixels / 255.]).clip(0,1))
                return IUV_image.pixels_with_channels_at_back(), mesh_instance.points

            def __init__(self, batch_size=BATCH_SIZE):
                self.batch_size = batch_size
                self.size = 10000 
                super().__init__()

            def __len__(self):
                return self.size // BATCH_SIZE

            def __getitem__(self, idx):
                batch_UV = []
                batch_mesh = []
                for _ in range(self.batch_size):
                    uv, mesh = self.random_sample_UV()
                    batch_UV.append(uv)
                    batch_mesh.append(mesh)

                batch_UV = np.array(batch_UV)
                batch_mesh = np.array(batch_mesh)

                return [batch_UV], [batch_mesh]

        return MeshRandomSample(batch_size=BATCH_SIZE)

    def build_model():
        input_uv = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_mesh')

        mesh_embedding = dm.networks.Encoder2D(
            input_uv, EMBEDING, depth=4, nf=64)
        output_mesh = dm.networks.MeshDecoder(
            mesh_embedding, 3, graph_laplacians, adj_matrices, upsamling_matrices, polynomial_order=6, filter_list=[16, 16, 16, 16])
        
        # wrapping input and output
        mesh_ae = dm.DeepMachine(
            inputs=input_uv, 
            outputs=[output_mesh]
        )
        
        # compile model with optimizer
        mesh_ae.compile(
            optimizer=dm.optimizers.RMSprop(lr=LR),
            loss=['mae']
        )

        
        return mesh_ae

    def custom_summary(train_x, train_y, predict_y):

        def render_mesh(sample_mesh, res=256, scale=MESH_SCALE):
            sample_mesh = Homogeneous(dm.utils.rotation_matrix(np.deg2rad(180),[0,1,0])).apply(sample_mesh)
            sample_mesh = ColouredTriMesh(
                sample_mesh * scale * res / 2 + res / 2,
                trilist=trilist,
                colours=np.ones_like(sample_mesh) * [0, 0, 1]
            )
            sample_mesh = lambertian_shading(sample_mesh, ambient_colour=0)
            m3io.export_mesh(sample_mesh, Path(LOGDIR)/'{}.obj'.format(time.time()))

            mesh_img = rasterize_mesh(
                sample_mesh,
                [res, res]
            )
            mesh_img = mesh_img.rotate_ccw_about_centre(90)

            return mesh_img.pixels_with_channels_at_back()

        return {
            'target/mesh': np.array(list(map(render_mesh, train_y[0]))),
            'output/mesh': np.array(list(map(render_mesh, predict_y[0]))),
         }

    training_generator = build_data()
    auto_encoder = build_model()

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
