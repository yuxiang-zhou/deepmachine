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
        class H5Data(dm.utils.Sequence):

            def __init__(self, fp, batch_size=BATCH_SIZE):
                self.train_data = h5py.File(fp, 'r')
                # self.train_data.swmr_mode = True
                self.batch_size = batch_size
                self.size = self.train_data['image'].len()
                self.indexes = list(range(self.size))
                np.random.shuffle(self.indexes)
                super().__init__()

            def __len__(self):
                return self.size // self.batch_size

            def __getitem__(self, idx):
                # training data
                batch_train_data = np.array([
                    # self.train_data['data'][self.indexes[i]]
                    np.concatenate([
                        self.train_data['image'][self.indexes[i]],
                        self.train_data['uv'][self.indexes[i]],
                        # dm.utils.lms_to_heatmap(self.train_data['lms'][self.indexes[i]], INPUT_SHAPE, INPUT_SHAPE).transpose([1,2,0])

                    ], axis=-1) 
                    for i in range(idx, idx+self.batch_size) if self.train_data['label'][self.indexes[i]][0] >= 0
                ])

                # testing data
                batch_label = np.array([
                    dm.utils.one_hot(self.train_data['label'][self.indexes[i]], n_parts=N_CLASSES) for i in range(idx, idx+self.batch_size) if self.train_data['label'][self.indexes[i]][0] >= 0
                ]).squeeze()


                return [batch_train_data, batch_label], [batch_label, batch_label]

            def on_epoch_end(self, *args, **kwargs):
                np.random.shuffle(self.indexes)
                return super().on_epoch_end()


        return H5Data(FLAGS.dataset_path, batch_size=BATCH_SIZE)

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
        logdir=LOGDIR,
        lr_decay=0.99,
        verbose=2,
        workers=FLAGS.no_thread
    )


if __name__ == '__main__':
    main()
