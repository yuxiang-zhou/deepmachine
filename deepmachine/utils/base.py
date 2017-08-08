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


slim = tf.contrib.slim

ResizeMethod = tf.image.ResizeMethod


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    # RGB -> BGR
    image = tf.reverse(image, [False, False, True])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image


def tf_rotate_points(points, image, angle):

    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # center coordinates since rotation center is supposed to be in the image center
    points_centered = points - image_center

    rot_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(
        angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)])
    rot_matrix = tf.reshape(rot_matrix, shape=[2, 2])

    points_centered_rot = tf.matmul(rot_matrix, tf.transpose(points_centered))

    return tf.transpose(points_centered_rot) + image_center


def tf_lms_to_heatmap(lms, h, w, n_landmarks, marked_index, sigma=5):
    xs, ys = tf.meshgrid(tf.range(0., tf.to_float(w)),
                         tf.range(0., tf.to_float(h)))
    gaussian = (1. / (sigma * np.sqrt(2. * np.pi)))

    def gaussian_fn(lms):
        y, x, idx = tf.unstack(lms)
        idx = tf.to_int32(idx)

        def run_true():
            return tf.exp(-0.5 * (tf.pow(ys - y, 2) + tf.pow(xs - x, 2)) *
                          tf.pow(1. / sigma, 2.)) * gaussian * 17.

        def run_false():
            return tf.zeros((h, w))

        return tf.cond(tf.reduce_any(tf.equal(marked_index, idx)), run_true, run_false)

    img_hm = tf.stack(tf.map_fn(gaussian_fn, tf.concat(
        [lms, tf.to_float(tf.range(0, n_landmarks))[..., None]], 1)))

    return img_hm


def tf_heatmap_to_lms(heatmap):
    hs = tf.argmax(tf.reduce_max(heatmap, 2), 1)
    ws = tf.argmax(tf.reduce_max(heatmap, 1), 1)
    lms = tf.transpose(tf.to_float(tf.stack([hs, ws])), perm=[1, 2, 0])

    return lms


def tf_image_batch_to_grid(images, col_size=4):
    image_shape = tf.shape(images)

    batch_size = image_shape[0]
    image_height = image_shape[1]
    image_width = image_shape[2]
    image_channels = image_shape[3]

    w = col_size
    h = batch_size // w

    tfimg = images[:w * h]
    tfimg = tf.reshape(
        tfimg, [w, h * image_height, image_width, image_channels])
    tfimg = tf.reshape(
        tf.transpose(tfimg, [1, 0, 2, 3]), [h * image_height, w * image_width, image_channels])

    return tfimg
