import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
import scipy

from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation

import sys

from ..flags import FLAGS
from .. import utils

slim = tf.contrib.slim

ResizeMethod = tf.image.ResizeMethod

class EarProvider(object):
    def __init__(self, filename=FLAGS.dataset_dir, batch_size=1, augmentation=False):
        self.filename = filename
        self.batch_size = batch_size
        self.augmentation = augmentation


    def get(self):
        images, *names = self._get_data_protobuff(self.filename)
        tensors = [images]

        for name in names:
            tensors.append(name)

        return tf.train.shuffle_batch(
            tensors, self.batch_size, 1000, 200, 4)

    def augmentation_type(self):
        return tf.stack([tf.random_uniform([1]),
                        (tf.random_uniform([1]) * 60. - 30.) * np.pi / 180.,
                        tf.random_uniform([1]) * 0.3 + 0.9])

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                # images
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'n_image': tf.FixedLenFeature([], tf.int64),
                'id_no': tf.FixedLenFeature([], tf.int64),
            }

        )
        return features

    def _image_from_feature(self, features):
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        n_image = tf.to_int32(features['n_image'])

        self.n_image = n_image
        #
        image = tf.reshape(image, (n_image, image_height, image_width, 3))
        image = tf.to_float(image) * 255.
        image = tf.random_shuffle(image)
        return image[:self.batch_size,...], image_height, image_width

    def _set_shape(self, image):
        image.set_shape([self.batch_size, 224, 224, 3])


    # Data from protobuff
    def _get_data_protobuff(self, filename):
        filename = str(filename).split(',')
        filename_queue = tf.train.string_input_producer(filename,
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self._get_features(serialized_example)

        # image
        image, image_height, image_width = self._image_from_feature(features)



        # augmentation
        if self.augmentation:
            do_flip, do_rotate, do_scale = tf.unstack(self.augmentation_type())

            # rescale
            image_height = tf.to_int32(tf.to_float(image_height) * do_scale[0])
            image_width = tf.to_int32(tf.to_float(image_width) * do_scale[0])

            image = tf.image.resize_images(image, tf.stack([image_height, image_width]))

            # rotate
            image = tf.contrib.image.rotate(image, do_rotate)

            # flip
            image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), image)


        # crop to 224 * 224
        target_h = tf.to_int32(224)
        target_w = tf.to_int32(224)
        offset_h = tf.to_int32((image_height - target_h) / 2)
        offset_w = tf.to_int32((image_width - target_w) / 2)

        image = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(
            img,offset_h, offset_w, target_h, target_w),
            image)


        self._set_shape(image)

        return image, []



class FaceProvider(object):
    def __init__(self, filename=FLAGS.dataset_dir, batch_size=1, id_size=1, augmentation=False):
        self.filename = filename
        self.batch_size = batch_size
        self.id_size = id_size
        self.augmentation = augmentation


    def get(self):
        images, *names = self._get_data_protobuff(self.filename)
        tensors = [images]

        for name in names:
            tensors.append(name)

        return tf.train.shuffle_batch(
            tensors, self.batch_size, 1000, 200, 4)

    def augmentation_type(self):
        return tf.stack([tf.random_uniform([1]),
                        (tf.random_uniform([1]) * 30. - 15.) * np.pi / 180.,
                        tf.random_uniform([1]) * 0.2 + 0.9])

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                # images
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'n_image': tf.FixedLenFeature([], tf.int64),
                'id_no': tf.FixedLenFeature([], tf.int64),
            }

        )
        return features

    def _image_from_feature(self, features):
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        n_image = tf.to_int32(features['n_image'])
        id_no = tf.to_int32(features['id_no'])

        self.n_image = n_image
        #
        image = tf.to_float(image)
        image = tf.image.per_image_standardization(image * 255.)
        image = tf.reshape(image, (n_image, image_height, image_width, 3))
        image = tf.random_shuffle(image)
        return image[:self.id_size,...], image_height, image_width, tf.tile([id_no-1], [self.id_size])

    def _set_shape(self, image):
        image.set_shape([self.id_size, FLAGS.image_size, FLAGS.image_size, 3])


    # Data from protobuff
    def _get_data_protobuff(self, filename):
        filename = str(filename).split(',')
        filename_queue = tf.train.string_input_producer(filename,
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self._get_features(serialized_example)

        # image
        image, image_height, image_width, id_no = self._image_from_feature(features)



        # augmentation
        if self.augmentation:
            do_flip, do_rotate, do_scale = tf.unstack(self.augmentation_type())

            # rescale
            # image_height = tf.to_int32(tf.to_float(image_height) * do_scale[0])
            # image_width = tf.to_int32(tf.to_float(image_width) * do_scale[0])

            image = tf.image.resize_images(image, tf.stack([image_height, image_width]))

            # rotate
            image = tf.contrib.image.rotate(image, do_rotate)

            # flip
            image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), image)


        # crop to 224 * 224
        target_h = tf.to_int32(FLAGS.image_size)
        target_w = tf.to_int32(FLAGS.image_size)
        offset_h = tf.to_int32((image_height - target_h) / 2)
        offset_w = tf.to_int32((image_width - target_w) / 2)

        image = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(
            img,offset_h, offset_w, target_h, target_w),
            image)


        self._set_shape(image)

        return image, id_no


# facenet data reader
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import os
from scipy import misc

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths):


    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_databatch(paths, n_ids):

    databatch = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                databatch += list(np.random.choice(image_paths, n_ids))

    return databatch



class ImageBatchLabelProvider(object):
    def __init__(self, dirpath=FLAGS.dataset_dir, batch_size=1, id_size=1, augmentation=False, image_size=160):
        self.dirpath = dirpath
        self.batch_size = batch_size
        self.id_size = id_size
        self.augmentation = augmentation
        self.image_size = image_size


    def get(self):
        databatch = get_databatch(self.dirpath, self.id_size)

        # image_paths_flat = []
        # labels_flat = []
        # for i in range(len(dataset)):
        #     image_paths_flat += dataset[i].image_paths
        #     labels_flat += [i] * len(dataset[i].image_paths)
        #
        # image_list = tf.convert_to_tensor(image_paths_flat)
        databatch_list = tf.convert_to_tensor(databatch)

        n_id = len(databatch) // self.id_size

        print('Number of identities: %d'%n_id)

        producer = tf.train.range_input_producer(n_id, shuffle=True)

        no_processes = 8
        images_batch = []
        labels_batch = []
        for _ in range(no_processes):
            index = producer.dequeue()
            image_paths = databatch_list[index*self.id_size:(index+1)*self.id_size]

            images = tf.map_fn(lambda x: tf.image.decode_png(tf.read_file(x)), image_paths, dtype=tf.uint8)

            if self.augmentation:
                # random rotate
                images = tf.map_fn(lambda img:tf.py_func(random_rotate_image, [img], tf.uint8), images)
                # random crop
                images = tf.random_crop(images, [self.id_size, self.image_size, self.image_size, 3])
                # random flip
                images = tf.map_fn(tf.image.random_flip_left_right, images)
            else:
                images = tf.map_fn(lambda img:tf.image.resize_image_with_crop_or_pad(img, self.image_size, self.image_size), images)

            images.set_shape((self.id_size, self.image_size, self.image_size, 3))
            images = tf.to_float(images)

            images = tf.map_fn(tf.image.per_image_standardization, images)
            labels = tf.tile([index], [self.id_size])

            images_batch.append(images)
            labels_batch.append(labels)

        return tf.train.batch(
            [images_batch, labels_batch], batch_size=self.batch_size, enqueue_many=True,
            capacity=4 * no_processes * self.batch_size,
            allow_smaller_final_batch=True)


class ImageLabelProvider(object):
    def __init__(self, dirpath=FLAGS.dataset_dir, batch_size=1, augmentation=False, image_size=160):
        self.dirpath = dirpath
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.image_size = image_size


    def get(self):

        dataset = []
        for path in self.dirpath.split(','):
            dataset += get_dataset(path)

        image_paths_flat = []
        labels_flat = []

        n_classes = len(dataset)

        for i in range(n_classes):
            image_paths_flat += dataset[i].image_paths
            labels_flat += [i] * len(dataset[i].image_paths)

        image_list = tf.convert_to_tensor(image_paths_flat)
        label_list = tf.convert_to_tensor(labels_flat)

        n_images = len(labels_flat)

        self.n_images = n_images
        self.n_classes = n_classes
        print('Number of Identities: %d, Number of images: %d'%(n_classes, n_images))

        producer = tf.train.range_input_producer(n_images, shuffle=True)

        no_processes = 8
        images = []
        labels = []
        for _ in range(no_processes):
            index = producer.dequeue()

            file_contents = tf.read_file(image_list[index])
            image = tf.image.decode_png(file_contents)
            if self.augmentation:
                # random rotate
                image = tf.py_func(random_rotate_image, [image], tf.uint8)
                # random crop
                image = tf.random_crop(image, [self.image_size, self.image_size, 3])
                # random flip
                image = tf.image.random_flip_left_right(image)
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, self.image_size, self.image_size)

            image.set_shape((self.image_size, self.image_size, 3))
            image = tf.to_float(image)

            images.append(tf.image.per_image_standardization(image))
            labels.append(label_list[index])

            # tensors = [image, label_list[index]]
            # image_label.append(tensors)

        return tf.train.batch(
            [images, labels], batch_size=self.batch_size, enqueue_many=True,
            capacity=4 * no_processes * self.batch_size,
            allow_smaller_final_batch=True)
