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

# deepmachine
import keras
import tensorflow as tf
import deepmachine as dm

# flag definitions
from deepmachine.flags import FLAGS


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

        dataset = dm.data.provider.TFRecordNoFlipProvider(
            DATA_PATH,
            dm.data.provider.features.FeatureIUVHM,
            augmentation=True,
            resolvers={
                'images': dm.data.provider.resolvers.image_resolver,
                'iuvs': partial(dm.data.provider.resolvers.iuv_resolver, n_parts=25, dtype=tf.float32),
                'heatmaps': partial(dm.data.provider.resolvers.heatmap_resolver, n_lms=17),
            }
        )
        dataset = dm.data.provider.DatasetQueue(
            dataset, n_proccess=FLAGS.no_thread, batch_size=BATCH_SIZE)
        tf_data = dataset.get('images', 'iuvs', 'heatmaps')

        return [tf_data['images']], [tf_data['iuvs'], tf_data['heatmaps'],tf_data['iuvs'], tf_data['heatmaps']]

    def model_builder():
        input_image = dm.layers.Input(
            shape=[INPUT_SHAPE, INPUT_SHAPE, 3], name='input_image')
        iuv_prediction = dm.networks.Hourglass(
            input_image, [256, 256, 75], nf=64, batch_norm='InstanceNormalization2D')

        hm_prediction = dm.networks.Hourglass(
            input_image, [256, 256, 17], nf=64, batch_norm='InstanceNormalization2D')

        merged_inputs = dm.layers.Concatenate()(
            [input_image, hm_prediction, iuv_prediction])

        iuv_prediction_refine = dm.networks.Hourglass(
            merged_inputs, [256, 256, 75], nf=64, batch_norm='InstanceNormalization2D')

        hm_prediction_refine = dm.networks.Hourglass(
            merged_inputs, [256, 256, 17], nf=64, batch_norm='InstanceNormalization2D')

        train_model = dm.DeepMachine(
            inputs=input_image, outputs=[
                iuv_prediction, hm_prediction, iuv_prediction_refine, hm_prediction_refine
            ])
        train_model.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=[
                dm.losses.loss_iuv_regression,
                dm.losses.loss_heatmap_regression,
                dm.losses.loss_iuv_regression,
                dm.losses.loss_heatmap_regression
            ],
            loss_weights=[1,1,1,1]
        )

        return train_model

    def summary_op(train_x, train_y, predict_y):

        return {
            'target/iuv': np.array(list(map(dm.utils.iuv_rgb, train_y[0]))),
            'target/heatmap': np.array(list(map(dm.utils.channels_to_rgb, train_y[1]))),
            'output/iuv_0': np.array(list(map(dm.utils.iuv_rgb, predict_y[0]))),
            'output/iuv_1': np.array(list(map(dm.utils.iuv_rgb, predict_y[2]))),
            'output/heatmap_0': np.array(list(map(dm.utils.channels_to_rgb, predict_y[1]))),
            'output/heatmap_1': np.array(list(map(dm.utils.channels_to_rgb, predict_y[3]))),
        }

    # ### Training

    model_builder().fit(
        build_data(),
        epochs=200, step_per_epoch=15000 // BATCH_SIZE,
        logdir=LOGDIR,
        verbose=2,
        summary_ops=[summary_op],
        lr_decay=0.98
    )


if __name__ == '__main__':
    main()
