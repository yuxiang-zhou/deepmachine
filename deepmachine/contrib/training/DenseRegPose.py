import os, shutil, math, time
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
    INPUT_SHAPE=256
    LR=FLAGS.lr
    DATA_PATH = '/vol/atlas/homes/yz4009/databases/tfrecords/densepose-train.tfrecord'
    LOGDIR = "/homes/yz4009/db/ckpt_all/Pose/densereg_keras/model_{}".format(time.time())


    def parse_record(example, aug=False):
        features = tf.parse_single_example(
            example,
            features=dm.data_provider.features.FeatureIUVHM
        )
        
        rand_aug = [
            0, # flip
            tf.random_uniform([1],)[0] * np.pi/2. - np.pi/4., # rotate
            tf.random_uniform([1],)[0] * 0.4 + 0.8, # scale
            tf.random_uniform([1],)[0] * 0.2, # offset_h
            tf.random_uniform([1],)[0] * 0.2, # offset_w
        ]
        
        images = dm.data_provider.resolvers.image_resolver(features, aug=aug, aug_args=rand_aug)
        iuvs = dm.data_provider.resolvers.iuv_resolver(features, aug=aug, aug_args=rand_aug,n_parts=25, dtype=tf.float32)
        heatmaps = dm.data_provider.resolvers.heatmap_resolver(features, aug=aug, aug_args=rand_aug, n_lms=17)
        return images, iuvs, heatmaps

    def model_builder():
        input_image = dm.layers.Input(shape=[INPUT_SHAPE,INPUT_SHAPE,3], name='input_image')
        iuv_prediction = dm.networks.Hourglass(input_image, [256,256,75], depth=4, batch_norm=True, use_coordconv=True)
        merged_inputs = dm.layers.Concatenate()([input_image, iuv_prediction])
        hm_prediction = dm.networks.Hourglass(merged_inputs, [256,256,17], depth=4, batch_norm=True, use_coordconv=True)
        
        train_model = dm.Model(inputs=input_image, outputs=[iuv_prediction, hm_prediction])
        
        return train_model

    def summary_op(train_x, train_y, predict_y):
            
        return {
            'target/iuv': np.array(list(map(dm.utils.iuv_rgb, train_y[0]))),
            'target/heatmap': np.array(list(map(dm.utils.channels_to_rgb,train_y[1]))),
            'output/iuv': np.array(list(map(dm.utils.iuv_rgb,predict_y[0]))),
            'output/heatmap': np.array(list(map(dm.utils.channels_to_rgb,predict_y[1]))),
        }

    train_dataset = tf.data.TFRecordDataset(DATA_PATH)

    train_dataset = train_dataset.shuffle(BATCH_SIZE * 10)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(partial(parse_record, aug=True), num_parallel_calls=4)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.contrib.data.AUTOTUNE)


    train_db_it = train_dataset.make_one_shot_iterator()
    train_x,train_y1,train_y2 = train_db_it.get_next()
    train_x.set_shape([BATCH_SIZE, INPUT_SHAPE, INPUT_SHAPE, 3])
    train_y1.set_shape([BATCH_SIZE, INPUT_SHAPE, INPUT_SHAPE, 75])
    train_y2.set_shape([BATCH_SIZE, INPUT_SHAPE, INPUT_SHAPE, 17])



    # test_dataset = tf.data.TFRecordDataset('/vol/atlas/homes/yz4009/databases/tfrecords/val_mpii_all.tfrecords')
    # test_dataset = test_dataset.shuffle(BATCH_SIZE * 10)
    # test_dataset = test_dataset.repeat()
    # test_dataset = test_dataset.map(parse_record, num_parallel_calls=1)
    # test_dataset = test_dataset.batch(BATCH_SIZE)
    # test_dataset = test_dataset.prefetch(tf.contrib.data.AUTOTUNE)


    # test_db_it = test_dataset.make_one_shot_iterator()
    # test_x,test_y1,test_y2 = test_db_it.get_next()
    # test_x.set_shape([BATCH_SIZE, 256, 256, 3])
    # test_y1.set_shape([BATCH_SIZE, 256, 256, 75])
    # test_y2.set_shape([BATCH_SIZE, 256, 256, 16])


    # ### Training

    DenseRegCascadeModel = dm.DeepMachine(
        network=model_builder(),
        ckpt_path=LOGDIR
    )

    DenseRegCascadeModel.compile(
        optimizer=dm.optimizers.Adam(lr=LR),
        loss=[dm.losses.loss_iuv_regression, dm.losses.loss_heatmap_regression],
    )

    DenseRegCascadeModel.fit_tf_data(
        [[train_x], [train_y1, train_y2]],
        epochs=200, step_per_epoch=15000 // BATCH_SIZE,
        logdir=LOGDIR,
        verbose=2,
        summary_ops=[summary_op],
        lr_decay=0.99
    )


if __name__ == '__main__':
    main()