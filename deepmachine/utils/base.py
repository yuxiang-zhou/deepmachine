import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
from scipy.interpolate import interp1d
import scipy as sp

from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation, Scale


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


def crop_image_bounding_box(img, bbox, res, base=200., order=1):

    center = bbox.centre()
    bmin, bmax = bbox.bounds()
    scale = np.linalg.norm(bmax - bmin) / base

    return crop_image(img, center, scale, res, order=order)


def crop_image(img, center, scale, res, base=384., order=1):
    h = base * scale

    t = Translation(
        [
            res[0] * (-center[0] / h + .5),
            res[1] * (-center[1] / h + .5)
        ]).compose_after(Scale((res[0] / h, res[1] / h))).pseudoinverse()

    # Upper left point
    ul = np.floor(t.apply([0, 0]))
    # Bottom right point
    br = np.ceil(t.apply(res).astype(np.int))

    # crop and rescale

    cimg, trans = img.warp_to_shape(
        br - ul, Translation(-(br - ul) / 2 + (br + ul) / 2), return_transform=True)
    c_scale = np.min(cimg.shape) / np.mean(res)
    new_img = cimg.rescale(1 / c_scale, order=order).resize(res, order=order)

    trans = trans.compose_after(Scale([c_scale, c_scale]))

    return new_img, trans, c_scale


def normalized_point_to_point_error(preds, gts, factor=1):
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2),
                                  reduction_indices=-1)) / factor
    return dists


def pckh(preds, gts, scales):
    t_range = np.arange(0, 0.51, 0.01)
    dists = normalized_point_to_point_error(preds, gts, factor=scales)
    return ced_accuracy(0.5, dists)


def ced_accuracy(t, dists):
    # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
    pts_r = tf.transpose(
        tf.gather(tf.transpose(dists), [8, 12, 11, 10, 2, 1, 0]))
    pts_l = tf.transpose(
        tf.gather(tf.transpose(dists), [9, 13, 14, 15, 3, 4, 5]))
    part_pckh = (tf.to_int32(pts_r <= t) + tf.to_int32(pts_l <= t)) / 2

    return tf.concat([part_pckh, tf.reduce_sum(tf.to_int32(dists <= t), 1)[..., None] / tf.shape(dists)[1]], 1)

def arclen_polyl(cnt):

    tang = np.diff(cnt, axis=0)
    seg_len = np.sqrt(np.power(tang[:, 0], 2) + np.power(tang[:, 1], 2))
    seg_len = np.hstack((0, seg_len))
    alparam = np.cumsum(seg_len)
    cntLen = alparam[-1]
    return alparam, cntLen

def interpolate(points, step, kind='slinear'):
    alparam, cntLen = arclen_polyl(points)

    f_x = interp1d(
        alparam, points[:, 0], kind=kind
    )

    f_y = interp1d(
        alparam, points[:, 1], kind=kind
    )

    points_dense_x = f_x(np.arange(0, cntLen, step))
    points_dense_y = f_y(np.arange(0, cntLen, step))

    points_dense = np.hstack((
        points_dense_x[:, None], points_dense_y[:, None]
    ))

    return points_dense

def multi_channel_svs(svs_pts, h,w, groups,c=3):
    msvs = Image.init_blank((h,w), n_channels=len(groups))
    for ch,g in enumerate(groups):
        if len(g):
            msvs.pixels[ch, ... ] = svs_shape(svs_pts, h,w, groups=[g],c=c).pixels[0]
    msvs.pixels /= np.max(msvs.pixels)
    return msvs

def svs_shape(pc, xr, yr, groups=None, c=1):
    store_image = Image.init_blank((xr,yr))
    ni = binary_shape(pc, xr, yr, groups)
    store_image.pixels[0,:,:] = sp.ndimage.filters.gaussian_filter(np.squeeze(ni.pixels), c)
    return store_image

def binary_shape(pc, xr, yr, groups=None):
    return sample_points(pc.points, xr, yr, groups)

def sample_points(target, range_x, range_y, edge=None, x=0, y=0):
    ret_img = Image.init_blank((range_x, range_y))

    if edge is None:
        for pts in target:
            ret_img.pixels[0, pts[0]-y, pts[1]-x] = 1.
    else:
        for eg in edge:
            for pts in interpolate(target[eg], 0.1):
                try:
                    ret_img.pixels[0, int(pts[0]-y), int(pts[1]-x)] = 1.
                except:
                    pass
                    # print('Index out of Bound')

    return ret_img
