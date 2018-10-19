# basic library
import os
import shutil
import math
import time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
from pathlib import Path
from functools import partial
from menpo3d.unwrap import optimal_cylindrical_unwrap

# deepmachine
import keras
import tensorflow as tf
import deepmachine as dm

# flag definitions
from deepmachine.flags import FLAGS

def main():
    BATCH_SIZE = FLAGS.batch_size
    INPUT_SHAPE=256
    LR=FLAGS.lr
    LOGDIR = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(FLAGS.logdir, int(time.time()))

    def build_data():
        # ### load models
        shape_model = mio.import_pickle('/homes/yz4009/wd/notebooks/Projects/MLProjects/models/all_all_all.pkl')
        uv_shape = (INPUT_SHAPE, INPUT_SHAPE)
        template = shape_model.instance([])
        UV_template = optimal_cylindrical_unwrap(template).apply(template)
        UV_template.points = UV_template.points[:, [1, 0]]
        UV_template.points[:, 0] = UV_template.points[:, 0].max() - UV_template.points[:, 0]

        UV_template.points -= UV_template.points.min(axis=0)
        UV_template.points /= UV_template.points.max(axis=0)
        UV_template.points *= np.array([uv_shape])


        # ### Generate IUV

        def rotation_matrix(angle, direction, point=None):
            sina = math.sin(angle)
            cosa = math.cos(angle)
            direction = np.array(direction).astype(np.float64)
            # rotation matrix around unit vector
            R = np.diag([cosa, cosa, cosa])
            R += np.outer(direction, direction) * (1.0 - cosa)
            direction *= sina
            R += np.array([[ 0.0,         -direction[2],  direction[1]],
                            [ direction[2], 0.0,          -direction[0]],
                            [-direction[1], direction[0],  0.0]])
            M = np.identity(4)
            M[:3, :3] = R
            if point is not None:
                # rotation not around origin
                point = np.array(point[:3], dtype=np.float64, copy=False)
                M[:3, 3] = point - np.dot(R, point)
            return M

        def random_sample_UV(n_pcs = 50, random_angle = 45, shape = [INPUT_SHAPE, INPUT_SHAPE]):
            # random mesh
            sample_mesh = shape_model.instance(np.random.sample([n_pcs]) * 3, normalized_weights=True)
            sample_mesh = ColouredTriMesh(sample_mesh.points, trilist=sample_mesh.trilist)
            sample_mesh = lambertian_shading(sample_mesh)
            sample_mesh = Homogeneous(rotation_matrix(np.deg2rad(90), [0,0,1])).apply(sample_mesh)
            sample_mesh = Homogeneous(rotation_matrix(np.deg2rad(180), [1,0,0])).apply(sample_mesh)
            sample_mesh = Scale([100,100,100]).apply(sample_mesh)

            # random rotation and scale and translation
            sample_mesh = Homogeneous(rotation_matrix(np.deg2rad(np.random.rand() * random_angle - random_angle / 2), [0,0,1])).apply(sample_mesh)
            sample_mesh = Homogeneous(rotation_matrix(np.deg2rad(np.random.rand() * random_angle - random_angle / 2), [0,1,0])).apply(sample_mesh)
            sample_mesh = Homogeneous(rotation_matrix(np.deg2rad(np.random.rand() * random_angle - random_angle / 2), [1,0,0])).apply(sample_mesh)

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
            return IUV_image

        def one_hot(a):
            a = a.astype(np.int32)
            n_parts = np.max(a) + 1
            b = np.zeros((len(a), n_parts))
            b[np.arange(len(a)), a] = 1
            return b

        def rgb_iuv(rgb):
            # formation
            iuv_mask = rgb[..., 0]
            n_parts = int(np.max(iuv_mask) + 1)
            iuv_one_hot = one_hot(iuv_mask.flatten()).reshape(iuv_mask.shape + (n_parts,))

            # normalised uv
            uv = rgb[..., 1:] / 255. if np.max(rgb[..., 1:]) > 1 else rgb[..., 1:]
            u = iuv_one_hot * uv[..., 0][..., None]
            v = iuv_one_hot * uv[..., 1][..., None]

            iuv = np.concatenate([iuv_one_hot, u, v], 2)

            return iuv

        class FaceIUVSequence(dm.utils.Sequence):

            def __init__(self, batch_size=16, is_validation=False, validation_size=BATCH_SIZE):
                self.batch_size = validation_size if is_validation else batch_size
                self.size = validation_size if is_validation else 10000 // BATCH_SIZE
                super().__init__()

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                batch_img = [
                    rgb_iuv(random_sample_UV(shape = [256, 256]).pixels_with_channels_at_back()) for _ in range(self.batch_size)
                ]
                
                batch_x = batch_y = batch_img = np.array(batch_img)
                return [batch_x], [batch_y]

        return FaceIUVSequence(batch_size=BATCH_SIZE)

    def build_model():
        input_image = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,6], name='input_image')
        ae_image, [vae_encoder, vae_decoder] = dm.networks.VAE(input_image, nf=64, depth=4, embedding=1024, latent=512, return_models=True)
        z_mean, z_log_var, _ = vae_encoder.outputs

        autoencoder_union = dm.DeepMachine(inputs=[input_image], outputs=[ae_image])

        def vae_loss(y_true, y_pred):
            reconstruction_loss = dm.losses.mse(y_true, y_pred)
            kl_loss = dm.losses.loss_kl(z_mean, z_log_var)
            return dm.K.mean(reconstruction_loss) + kl_loss

        autoencoder_union.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=vae_loss
        )
        return autoencoder_union

    training_generator = build_data()
    auto_encoder = build_model()

    results = auto_encoder.fit(
        training_generator, 
        epochs=200,
        lr_decay=0.99,
        logdir=LOGDIR,
        verbose=2,
        workers=FLAGS.no_thread
    )

if __name__ == '__main__':
    main()