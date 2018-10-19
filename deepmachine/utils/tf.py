import keras
import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
from scipy.interpolate import interp1d
import scipy as sp
from keras import backend as K
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation, Scale

from .np import sample_colours_from_colourmap

ResizeMethod = tf.image.ResizeMethod

# tf functions


def tf_caffe_preprocess(image):
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


def tf_image_patch_around_lms(image, lms, patch_size=32, dtype=tf.float32):

    pad_size = patch_size // 2 + 1
    lms = tf.to_int32(lms) + tf.constant([pad_size, pad_size])
    image = tf.pad(image, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]])

    def crop(x):
        return tf.image.crop_to_bounding_box(image, x[0] - pad_size, x[1] - pad_size, patch_size, patch_size)

    image = tf.concat(tf.unstack(tf.map_fn(crop, lms, dtype=dtype)), axis=-1)

    return image


def tf_records_iterator(path, feature=None):

    record_iterator = tf.python_io.tf_record_iterator(path=path)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        yield example.features.feature


def tf_logits_to_heatmap(logits, num_classes):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, FLAGS.n_landmarks + 1].
    """

    keypoint_colours = np.array(
        [
            plt.cm.spectral(x)
            for x in np.linspace(0, 1, num_classes + 1)
        ])[..., :3].astype(np.float32)

    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(
        prediction, (-1, num_classes + 1)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap


def tf_logits_to_landmarks(keypoints):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    zeros = tf.to_float(tf.zeros_like(is_background))

    return tf.where(is_background, zeros, ones) * 255


def tf_keypts_encoding(keypoints, num_classes):
    keypoints = tf.to_int32(keypoints)
    keypoints = tf.reshape(keypoints, (-1,))
    keypoints = tf.layers.one_hot_encoding(
        keypoints, num_classes=num_classes + 1)
    return keypoints


def tf_get_weight(keypoints, mask=None, ng_w=0.01, ps_w=1.0):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    weights = tf.where(is_background, ones * ng_w, ones * ps_w)
    # if mask is not None:
    #     weights *= tf.to_float(mask)

    return weights


def tf_atan2(y, x):
    angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
    angle = tf.where(tf.greater(y, 0.0), 0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.where(tf.less(y, 0.0), -0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.where(tf.less(x, 0.0), tf.atan(y / x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)),
                     np.nan * tf.zeros_like(x), angle)

    indices = tf.where(tf.less(angle, 0.0))
    updated_values = tf.gather_nd(angle, indices) + (2 * np.pi)
    update = tf.SparseTensor(indices, updated_values, angle.get_shape())
    update_dense = tf.sparse_tensor_to_dense(update)

    return angle + update_dense


def tf_n_channel_rgb(inputs, n_feature, colour_set='jet'):
    cm = sample_colours_from_colourmap(
        n_feature, colour_set).astype(np.float32)
    tf_cm = tf.constant(cm)
    tf_img = tf.tensordot(inputs, tf_cm, axes=1)

    return tf_img


def tf_iuv_rgb(tf_iuv, n_feature=26, colour_set='jet'):

    tf_iuv_class = tf_iuv[..., :n_feature]
    tf_iuv_class = tf.argmax(tf_iuv_class, axis=-1)
    tf_iuv_class = tf.one_hot(tf_iuv_class, n_feature)

    tf_u = tf_iuv_class * tf_iuv[..., n_feature:n_feature*2]
    tf_v = tf_iuv_class * tf_iuv[..., n_feature*2:]

    tf_u = tf_n_channel_rgb(tf_u, n_feature)
    tf_v = tf_n_channel_rgb(tf_v, n_feature)

    tf_img = (tf_u + tf_v) / 2. / 255.

    return tf_img


def tf_ced_accuracy(t, dists):
    # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
    pts_r = tf.transpose(
        tf.gather(tf.transpose(dists), [8, 12, 11, 10, 2, 1, 0]))
    pts_l = tf.transpose(
        tf.gather(tf.transpose(dists), [9, 13, 14, 15, 3, 4, 5]))
    part_pckh = (tf.to_int32(pts_r <= t) + tf.to_int32(pts_l <= t)) / 2

    return tf.concat([part_pckh, tf.reduce_sum(tf.to_int32(dists <= t), 1)[..., None] / tf.shape(dists)[1]], 1)


def tf_normalized_point_to_point_error(preds, gts, factor=1):
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2),
                                  reduction_indices=-1)) / factor
    return dists

def tf_pckh(preds, gts, scales):
    t_range = np.arange(0, 0.51, 0.01)
    dists = tf_normalized_point_to_point_error(preds, gts, factor=scales)
    return tf_ced_accuracy(0.5, dists)