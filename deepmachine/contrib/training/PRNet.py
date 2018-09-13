import os
import shutil
import math
import time
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
import deepmachine as dm
import tensorflow as tf
import keras

from pathlib import Path
from functools import partial
from deepmachine.flags import FLAGS
from deepmachine import data_provider
from menpo.shape import PointCloud, TriMesh, ColouredTriMesh, PointUndirectedGraph
from menpo.visualize import print_progress
from menpo.image import Image
from menpo.transform import Translation
from menpo3d.camera import PerspectiveCamera
from menpo3d.unwrap import optimal_cylindrical_unwrap
from menpo3d.rasterize import rasterize_mesh


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
            dm.utils.union_dict([
                data_provider.features.image_feature,
                data_provider.features.uvxyz_feature
            ]),
            augmentation=False,
            resolvers={
                'images': data_provider.resolvers.image_resolver,
                'uvxyz': data_provider.resolvers.uvxyz_resolver
            }
        )
        dataset = data_provider.DatasetQueue(
            dataset, n_proccess=FLAGS.no_thread, batch_size=BATCH_SIZE)
        tf_data = dataset.get('images', 'uvxyz')

        return [tf_data['images']], [tf_data['uvxyz']]

    def model_builder():
        input_image = dm.layers.Input(
            shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_image')
        uvxyz_prediction = dm.networks.ResNet50(
            input_image, [256, 256, 3])
        
        train_model = dm.DeepMachine(
            inputs=input_image, outputs=[
                uvxyz_prediction])
        train_model.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=[
                'mae'
            ],
        )

        return train_model

    def summary_op(train_x, train_y, predict_y):

        return {
            
        }

    # ### Training

    model_builder().fit(
        build_data(),
        epochs=200, step_per_epoch=15000 // BATCH_SIZE,
        logdir=LOGDIR,
        verbose=2,
        summary_ops=[summary_op],
        lr_decay=0.99
    )


if __name__ == '__main__':
    main()
