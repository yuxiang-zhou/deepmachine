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

def format_folder(FLAGS):
    post_fix = 'lr{:.5f}_d{:.3f}_b{:03d}'.format(
        FLAGS.lr, FLAGS.lr_decay, FLAGS.batch_size
    )

    logdir = FLAGS.logdir if 'model_' in FLAGS.logdir else "{}/model_{}".format(
        FLAGS.logdir, post_fix
    )

    return logdir

def main():
    tf.reset_default_graph()
    BATCH_SIZE = FLAGS.batch_size
    INPUT_SHAPE = 112
    INPUT_CHANNELS = 3
    NF = 64
    N_CLASSES = 8631
    LR = FLAGS.lr
    LOGDIR = format_folder(FLAGS)

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

        # batch_input = tf.concat([
        #     tf_data['image'], tf_data['uv']
        # ], axis=-1)

        batch_input = tf_data['image']
        label = tf_data['label']
        label = tf.squeeze(label)

        return [batch_input, label], [label, label]

    # Model
    def build_model():
        input_image = dm.layers.Input(
            shape=[INPUT_SHAPE, INPUT_SHAPE, INPUT_CHANNELS], name='input_image')

        embeding, softmax = dm.networks.ArcFace(
            [input_image], 512, nf=NF, n_classes=N_CLASSES,
            batch_norm='BatchNormalization'
        )
        
        train_model = dm.DeepMachine(
            inputs=[input_image], outputs=[embeding, softmax])

        n_gpu = len(FLAGS.gpu.split(','))
        if n_gpu > 1:
            train_model = multi_gpu_model(train_model, gpus=n_gpu)

        def arc_loss(y_true, y_pred, s=64., m1=1., m2=0.3, m3=0.):
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

    arcface = build_model()
    def lr_sch_fn(epoch):
        new_lr = LR
        if epoch >= 15:
            new_lr /= 10.

        if epoch >= 22:
            new_lr /= 10.

        if epoch >= 26:
            new_lr /= 10.

        return new_lr 

    lr_sch = dm.callbacks.LearningRateScheduler(
        schedule=lr_sch_fn)
    lr_sch.set_model(arcface)

    arcface.fit(
        build_data(),
        epochs=30,
        step_per_epoch=N_CLASSES * 50 // BATCH_SIZE,
        logdir=LOGDIR,
        lr_decay=0,
        verbose=2,
        callbacks=[
            lr_sch
        ],
    )


if __name__ == '__main__':
    main()
