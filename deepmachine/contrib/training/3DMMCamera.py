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
from deepmachine.flags import FLAGS

def main():
    dm.K.clear_session()
    dm.K.set_learning_phase(1) #set learning phase

    # hyperparameters
    BATCH_SIZE = FLAGS.batch_size
    LR = FLAGS.lr
    LOGDIR = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
        FLAGS.logdir, int(time.time()))
    N_VERTICES = 28431
    EMBEDING = 128
    CAMERA_PARAM = 12
    INPUT_SHAPE = 112
    FILTERS = [16, 32, 32, 64]

    # globel constant
    n_gpu = len(FLAGS.gpu.split(','))
    face_mean_crop = m3io.import_mesh(FLAGS.meta_path + '/face_mean_mesh_crop.obj')
    trilist = face_mean_crop.trilist
    graph_laplacians, downsampling_matrices, upsamling_matrices, adj_matrices = mio.import_pickle(
        FLAGS.meta_path + '/mein3dcrop_LDUA.pkl', encoding='latin1')

    def build_data():
        
        class H5Mesh(dm.utils.Sequence):

            def __init__(self, fp, dataset, batch_size=BATCH_SIZE):
                self.train_mesh = h5py.File(fp, 'r')[dataset]
                self.batch_size = batch_size
                self.size = self.train_mesh.len()
                self.indexes = list(range(self.size))
                np.random.shuffle(self.indexes)
                super().__init__()

            def __len__(self):
                return self.size // self.batch_size

            def __getitem__(self, idx):
                indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_sample_mesh = np.array([
                    self.train_mesh[i] for i in indexes
                ])

                return [batch_sample_mesh], [batch_sample_mesh]

            def on_epoch_end(self, *args, **kwargs):
                np.random.shuffle(self.indexes)
                return super().on_epoch_end()

        class ImageSequence(dm.utils.Sequence):

            def __init__(self, dirpath, batch_size=BATCH_SIZE):
                self.detection = pd.read_csv('/homes/yz4009/db/face/loose_landmark_test.csv')
                self.size = self.detection.shape[0]
                self.image_path = Path(dirpath)
                self.batch_size = BATCH_SIZE
                self.indexes = list(range(self.size))
                np.random.shuffle(self.indexes)

            def on_epoch_end(self, *args, **kwargs):
                np.random.shuffle(self.indexes)

            def __len__(self):
                return self.size // self.batch_size

            def _preprocess(self, idx):
                name, *lms5pt = self.detection.loc[idx]
                lms5pt = PointCloud(np.array(lms5pt).reshape([-1,2])[:,::-1])
                img = mio.import_image((self.image_path/name).with_suffix('.jpg'))
                cimg, _, _ = dm.utils.crop_image_bounding_box(img, lms5pt.bounding_box(), [112, 112], base=186)

                return cimg.pixels_with_channels_at_back() * 2 - 1

            def __getitem__(self, idx):
                image_indexes = self.indexes[
                    idx * self.batch_size: (idx + 1) * self.batch_size]

                batch_img = [self._preprocess(i) for i in image_indexes]

                return [np.array(batch_img)], [np.array(batch_img)]
        
        return H5Mesh('/homes/yz4009/wd/gitdev/coma/data/mein3dcrop.h5', 'colour_mesh', batch_size=BATCH_SIZE), ImageSequence(FLAGS.dataset_path)

    def build_model(inputs_channels=6, n_gpu=n_gpu):

        # define components
        ## image encoder
        def build_img_encoder():
            input_img = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_img')

            img_embedding = dm.networks.Encoder2D(
                input_img, EMBEDING + CAMERA_PARAM, depth=4, nf=64)
            mesh_rec_embeding = dm.layers.Lambda(lambda x: x[..., :EMBEDING])(img_embedding)
            cam_rec_embeding = dm.layers.Lambda(lambda x: dm.K.tanh(x[..., EMBEDING:]) * 3)(img_embedding)

            return dm.Model(input_img, [mesh_rec_embeding, cam_rec_embeding], name='image_encoder')

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

        ## renderer
        def build_renderer(mesh_vertices, vertex_color, cam_parameter):
            # mesh_vertices = dm.layers.Input(shape=[N_VERTICES, 3], name='mesh_vertices')
            mesh_vertices.set_shape([BATCH_SIZE, N_VERTICES, 3])
            # vertex_color = dm.layers.Input(shape=[N_VERTICES, 3], name='vertex_color')
            vertex_color.set_shape([BATCH_SIZE, N_VERTICES, 3])
            # cam_parameter = dm.layers.Input(shape=[CAMERA_PARAM], name='cam_parameter')
            cam_parameter.set_shape([BATCH_SIZE, CAMERA_PARAM])

            # Build vertices and normals
            mesh_normals = tf.nn.l2_normalize(mesh_vertices, axis=2)

            # rendering output
            mesh_triangles = tf.constant(trilist, dtype=tf.int32)
            
            # camera position:
            eye = cam_parameter[...,:3]
            center = cam_parameter[...,3:6]
            world_up = cam_parameter[...,6:9]
            light_positions = cam_parameter[:,None,9:12]

            ambient_colors = tf.ones([BATCH_SIZE, 3], dtype=tf.float32) * 0.1
            light_intensities = tf.ones([BATCH_SIZE, 1, 3], dtype=tf.float32)

            render_mesh = dm.layers.Renderer(
                # image size
                image_width=INPUT_SHAPE,
                image_height=INPUT_SHAPE,
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

            render_mesh = dm.layers.Lambda(lambda x: x[..., :3])(render_mesh)

            return render_mesh

        # Mesh AE stream
        ## define inputs
        input_mesh_stream = dm.layers.Input(shape=[N_VERTICES, 6], name='input_mesh_stream')

        ## define components
        mesh_encoder_model = build_mesh_encoder()
        decoder_model = build_decoder()

        ## define connections
        output_mesh = decoder_model(mesh_encoder_model(input_mesh_stream))
        mesh_ae_model = dm.DeepMachine(
            inputs=input_mesh_stream, 
            outputs=output_mesh,
            name='MeshStream'
        )

        ## multi gpu support
        if n_gpu > 1:
            mesh_ae_model = multi_gpu_model(mesh_ae_model, gpus=n_gpu)

        ## compile mesh stream
        mesh_ae_model.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=['mae']
        )

        ## set trainable
        mesh_ae_model.trainable = False
        decoder_model.trainable = False
        mesh_encoder_model.trainable = False

        # Render Stream
        ## define inputs
        input_image_stream = dm.layers.Input(shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_image_stream')

        ## define components
        img_encoder_model = build_img_encoder()

        ## define connections
        rec_mesh_emb, rec_cam_emb = img_encoder_model(input_image_stream)
        mesh_with_colour = decoder_model(rec_mesh_emb)
        
        mesh_vert = dm.layers.Lambda(lambda x: x[..., :3])(mesh_with_colour)
        mesh_vert.set_shape([BATCH_SIZE, N_VERTICES, 3])
        mesh_colour = dm.layers.Lambda(lambda x: x[..., 3:])(mesh_with_colour)
        mesh_colour.set_shape([BATCH_SIZE, N_VERTICES, 3])
        rec_render = build_renderer(
            mesh_vert,
            mesh_colour,
            rec_cam_emb
        )

        render_model = dm.DeepMachine(
            inputs=input_image_stream, 
            outputs=[rec_render, mesh_with_colour],
            name='ImageStream'
        )
        
        ## multi gpu support
        if n_gpu > 1:
            render_model = multi_gpu_model(render_model, gpus=n_gpu)
        
        ## compile render stream
        render_model.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=['mae', dm.losses.dummy]
        )

        return render_model, mesh_ae_model, img_encoder_model

    def train_op(models, data, i_epoch, i_batch, epoch_end, training_history=None, **kwargs):
        sess = dm.K.get_session()
        image_stream, mesh_stream, img_encoder  = models
        [train_mesh, train_image], _ = dm.engine.training.generator_adapter(data)

        # ----------------------
        #  Train Mesh Stream
        # ----------------------
        loss_mesh = mesh_stream.train_on_batch([train_mesh], [train_mesh])

        # ------------------
        #  Train Render Stream
        # ------------------
        loss_img = image_stream.train_on_batch([train_image], [train_image, train_mesh])

        logs = dm.utils.Summary(
            {
                "losses/loss_mesh": loss_mesh,
                "losses/loss_img": loss_img[0],
                "learning_rate/mesh": mesh_stream.optimizer.lr.eval(sess),
                "learning_rate/img": image_stream.optimizer.lr.eval(sess),
            }
        )

        if epoch_end:
            ae_mesh = mesh_stream.predict(train_mesh)
            rec_imgs, rec_mesh = image_stream.predict(train_image)
            _, cam_params = img_encoder.predict(train_image)
            logs.update_images({
                'image/input': train_image,
                'image/render': rec_imgs,
                'image/mesh': dm.utils.mesh.render_meshes(rec_mesh[:4], trilist, res=INPUT_SHAPE),
                'mesh/input': dm.utils.mesh.render_meshes(train_mesh[:4], trilist, res=INPUT_SHAPE),
                'mesh/ae': dm.utils.mesh.render_meshes(ae_mesh[:4], trilist, res=INPUT_SHAPE),
            })

            logs.update_scalars(
                {'cam_params/{}'.format(idx_p): p for idx_p, p in enumerate(cam_params[0])}
            )

        return logs

    # prepare data
    train_generator = dm.data.generator.MergeGenerators(*build_data())
    train_queue = enqueue_generator(
        train_generator, workers=FLAGS.no_thread)
    
    # prepare model
    image_stream, mesh_stream, img_encoder = build_model()

    mesh_lr_decay = dm.callbacks.LearningRateScheduler(
        schedule=lambda epoch: LR * FLAGS.lr_decay ** epoch)
    mesh_lr_decay.set_model(mesh_stream)

    image_lr_decay = dm.callbacks.LearningRateScheduler(
        schedule=lambda epoch: LR * FLAGS.lr_decay ** epoch)
    image_lr_decay.set_model(image_stream)

    # training
    history = dm.engine.training.train_monitor(
        [image_stream, mesh_stream, img_encoder],
        train_queue, 
        train_op,
        epochs=200, step_per_epoch=len(train_generator),
        callbacks=[
            train_generator,
            mesh_lr_decay, image_lr_decay
        ],
        verbose=FLAGS.verbose,
        logdir=LOGDIR,
    )


if __name__ == '__main__':
    main()
