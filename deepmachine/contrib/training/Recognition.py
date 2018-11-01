# basic library
import os
import shutil
import math
import time
import h5py
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
from pathlib import Path
from functools import partial

# deepmachine
import keras
import tensorflow as tf
import deepmachine as dm
from deepmachine.utils.machine import multi_gpu_model
# flag definitions
from deepmachine.flags import FLAGS

def main():
    tf.reset_default_graph()
    BATCH_SIZE = FLAGS.batch_size
    INPUT_SHAPE = 112
    INPUT_CHANNELS = 5
    NF = 64
    N_CLASSES = 8631
    LR = FLAGS.lr
    LOGDIR = "{}/model_{}".format(FLAGS.logdir, time.time()
                                  ) if 'model_' not in FLAGS.logdir else FLAGS.logdir
    
    # Dataset
    def build_data():
        features = dm.utils.union_dict([
            dm.data.provider.features.image_feature(),
            dm.data.provider.features.tensor_feature('uv'),
            dm.data.provider.features.array_feature('label'),
            dm.data.provider.features.lms_feature('landmarks'),
        ])

        dataset = dm.data.provider.TFRecordProvider(
            FLAGS.dataset_path,
            features,
            resolvers={
                'image': partial(dm.data.provider.resolvers.image_resolver, output_shape=[INPUT_SHAPE, INPUT_SHAPE]),
                'uv': partial(dm.data.provider.resolvers.tensor_resolver, input_shape=[INPUT_SHAPE,INPUT_SHAPE, 2]),
                'landmarks': partial(dm.data.provider.resolvers.heatmap_resolver, n_lms=5, output_shape=[INPUT_SHAPE, INPUT_SHAPE]),
                'label': partial(dm.data.provider.resolvers.label_resolver, input_shape=[1], n_class=N_CLASSES),
            }
        )
        dataset = dm.data.provider.DatasetQueue(
            dataset, n_proccess=FLAGS.no_thread, batch_size=BATCH_SIZE)
        tf_data = dataset.get('image', 'uv', 'landmarks', 'label')

        batch_input = tf.concat([
            tf_data['image'], tf_data['uv']
        ], axis=-1)

        return [batch_input, tf_data['label']], [tf_data['label'], tf_data['label']]

    # Model
    def build_model():
        input_image = dm.layers.Input(
            shape=[INPUT_SHAPE, INPUT_SHAPE, INPUT_CHANNELS], name='input_image')
        input_label = dm.layers.Input(
            shape=[N_CLASSES], name='input_label')

        embeding, softmax = dm.networks.ArcFace(
            [input_image, input_label], 512, nf=NF, n_classes=N_CLASSES,
            batch_norm='BatchNormalization'
        )
        
        train_model = dm.DeepMachine(
            inputs=[input_image, input_label], outputs=[embeding, softmax])

        n_gpu = len(FLAGS.gpu.split(','))
        if n_gpu > 1:
            train_model = multi_gpu_model(train_model, gpus=n_gpu)

        def arc_loss(y_true, y_pred, s=64., m1=1., m2=0.35, m3=0.):
            # arc feature
            arc = y_pred * y_true
            arc = tf.acos(arc)
            arc = tf.cos(arc * m1 + m2) - m3
            arc = arc * s

            # softmax
            pred_softmax = dm.K.softmax(arc)
            return dm.losses.categorical_crossentropy(y_true, pred_softmax)


        train_model.compile(
            optimizer=dm.optimizers.Adam(lr=LR),
            loss=[dm.losses.dummy, arc_loss],
        )

        return train_model

    build_model().fit(
        build_data(),
        epochs=200,
        step_per_epoch=N_CLASSES * 50 // BATCH_SIZE,
        logdir=LOGDIR,
        lr_decay=0.99,
        verbose=2,
    )


if __name__ == '__main__':
    main()
