# coding: utf-8
import os
GPU="0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import shutil, math
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np
import deepmachine as dm
import tensorflow as tf

from functools import partial
from deepmachine import data_provider
from menpo.shape import PointCloud, TriMesh, ColouredTriMesh
from pathlib import Path
from menpo.visualize import print_progress
from menpo.image import Image
from menpo.transform import Translation
from menpo3d.camera import PerspectiveCamera
from menpo3d.unwrap import optimal_cylindrical_unwrap
from menpo3d.rasterize import rasterize_mesh

# ### TF Database Test

# In[4]:


BATCH_SIZE = 32

# In[6]:


def parse_record(example, aug=False):
    features = tf.parse_single_example(
        example,
        features=dm.data_provider.features.FeatureIUVHM
    )
    
    rand_aug = [0,0,tf.random_uniform([1],)[0] * 0.4 + 0.8,0,0]
    
    images = dm.data_provider.resolvers.image_resolver(features, aug=aug, aug_args=rand_aug)
    iuvs = dm.data_provider.resolvers.iuv_resolver(features, aug=aug, aug_args=rand_aug)
    heatmaps = dm.data_provider.resolvers.heatmap_resolver_pose(features, aug=aug, aug_args=rand_aug)
    return images, iuvs, heatmaps


# In[7]:
train_dataset = tf.data.TFRecordDataset('/data/yz4009/train_mpii_all.tfrecords')

train_dataset = train_dataset.apply(
    tf.contrib.data.shuffle_and_repeat(BATCH_SIZE * 10)
)
train_dataset = train_dataset.map(partial(parse_record, aug=True), num_parallel_calls=64)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.contrib.data.AUTOTUNE)


train_db_it = train_dataset.make_one_shot_iterator()
train_x,train_y1,train_y2 = train_db_it.get_next()


# In[8]:
test_dataset = tf.data.TFRecordDataset('/vol/atlas/homes/yz4009/databases/tfrecords/val_mpii_all.tfrecords')

test_dataset = test_dataset.apply(
    tf.contrib.data.shuffle_and_repeat(BATCH_SIZE * 10)
)
test_dataset = test_dataset.map(parse_record, num_parallel_calls=32)
test_dataset = test_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.apply(
#     tf.contrib.data.map_and_batch(parse_record, BATCH_SIZE, num_parallel_batches=16)
# )
test_dataset = test_dataset.prefetch(tf.contrib.data.AUTOTUNE)


test_db_it = test_dataset.make_one_shot_iterator()
test_x,test_y1,test_y2 = test_db_it.get_next()



# ### Training

# In[8]:


import time
logdir = "/homes/yz4009/db/ckpt_all/Pose/densereg_keras/model_{}".format(time.time())


# In[9]:


def model_builder():
    input_image = dm.layers.Input(shape=[256,256,3], name='input_image')
    iuv_prediction = dm.networks.Hourglass(input_image, [256,256,78], depth=4, batch_norm=False)
    merged_inputs = dm.layers.Concatenate()([input_image, iuv_prediction])
    hm_prediction = dm.networks.Hourglass(merged_inputs, [256,256,16], depth=4, batch_norm=False)

    train_model = dm.Model(inputs=input_image, outputs=[iuv_prediction, hm_prediction])
    
    return train_model


# In[10]:


DenseRegCascadeModel = dm.DeepMachine(
    network=model_builder(),
    ckpt_path=logdir
)


# In[12]:


DenseRegCascadeModel.compile(
    optimizer=dm.optimizers.Nadam(lr=1e-4),
    loss=[dm.losses.loss_iuv_regression, dm.losses.loss_heatmap_regression]
)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


results = DenseRegCascadeModel.fit(
    x=train_x,
    y=[train_y1,train_y2],
    epochs=200,
    steps_per_epoch=15000 // BATCH_SIZE,
    validation_data=[test_x, [test_y1,test_y2]],
    validation_steps=1000 // BATCH_SIZE,
)