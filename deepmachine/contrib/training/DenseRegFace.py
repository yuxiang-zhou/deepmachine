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
                'iuvs': partial(dm.data.provider.resolvers.iuv_resolver, n_parts=2, dtype=tf.float32),
                'heatmaps': dm.data.provider.resolvers.heatmap_resolver_face,
            }
        )
        dataset = dm.data.provider.DatasetQueue(
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
