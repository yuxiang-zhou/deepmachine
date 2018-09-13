import tensorflow as tf
import shutil
import math
import time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
import deepmachine as dm


from deepmachine.flags import FLAGS
from deepmachine import data_provider
from menpo.shape import PointCloud, TriMesh, ColouredTriMesh
from pathlib import Path
from menpo.visualize import print_progress
from menpo.image import Image
from menpo.transform import Translation
from menpo3d.camera import PerspectiveCamera
from menpo3d.unwrap import optimal_cylindrical_unwrap
from menpo3d.rasterize import rasterize_mesh
from functools import partial


def main():
    tf.reset_default_graph()
    BATCH_SIZE = FLAGS.batch_size
    INPUT_SHAPE = 256
    LR = FLAGS.lr
    DATA_PATH = FLAGS.dataset_path
    LOGDIR = "{}/model_{}".format(FLAGS.logdir, time.time()
                                  ) if 'model_' not in FLAGS.logdir else FLAGS.logdir
    # Dataset

    def build_data():
        dataset = data_provider.TFRecordNoFlipProvider(
            DATA_PATH,
            data_provider.features.FeatureIUVHM,
            augmentation=True,
            resolvers={
                'images': data_provider.resolvers.image_resolver,
                'iuvs': partial(data_provider.resolvers.iuv_resolver, n_parts=2, dtype=tf.float32),
                'heatmaps': data_provider.resolvers.heatmap_resolver_face,
            }
        )
        dataset = data_provider.DatasetQueue(
            dataset, n_proccess=FLAGS.no_thread, batch_size=BATCH_SIZE)
        tf_data = dataset.get('images', 'iuvs', 'heatmaps')

        return [tf_data['images']], [tf_data['iuvs'], tf_data['heatmaps']]

    # Model
    def build_model():
        input_image = dm.layers.Input(
            shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_image')

        iuv_prediction = dm.networks.Hourglass(
            input_image, [INPUT_SHAPE, INPUT_SHAPE, 6], depth=4, batch_norm=True, use_coordconv=False)
        merged_inputs = dm.layers.Concatenate()([input_image, iuv_prediction])
        hm_prediction = dm.networks.Hourglass(
            merged_inputs, [INPUT_SHAPE, INPUT_SHAPE, 68], depth=4, batch_norm=True, use_coordconv=False)

        train_model = dm.DeepMachine(
            inputs=input_image, outputs=[
                iuv_prediction, hm_prediction])

        train_model.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=[dm.losses.loss_iuv_regression,
                  dm.losses.loss_heatmap_regression],
        )

        return train_model

    build_model().fit(
        build_data(),
        epochs=200,
        step_per_epoch=40000 // BATCH_SIZE,
        logdir=LOGDIR,
        lr_decay=0.99,
        verbose=2
    )


if __name__ == '__main__':
    main()
