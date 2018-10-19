import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
import scipy
import functools

from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation

import sys
from ... import utils

ResizeMethod = tf.image.ResizeMethod


def dummy_resolver(_, *args, **kwargs):
    dummy = tf.constant(np.random.sample([1]).astype(np.float32))
    dummy.set_shape([1])

    return dummy


def dummy_seq_resolver(features, *args, **kwargs):
    frames = features['frames'].values
    n_data = tf.shape(frames)[0]
    window_size = 3

    dummy = tf.constant(np.random.sample([1]).astype(np.float32))
    dummy_sequences = tf.map_fn(lambda x: dummy, tf.range(
        n_data - window_size + 1), dtype=tf.float32)

    return dummy_sequences


def image_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0])):
    # load features
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])

    # formation
    image = tf.reshape(image, (image_height, image_width, 3))
    image = tf.to_float(image) / 255.

    # augmentation
    if aug:
        do_flip, do_rotate, do_scale, h_aug_offset, w_aug_offset = tf.unstack(aug_args)

        # scale
        image_height = tf.to_int32(tf.to_float(image_height) * do_scale)
        image_width = tf.to_int32(tf.to_float(image_width) * do_scale)

        image = tf.image.resize_images(
            image,
            tf.stack([image_height, image_width]),
            method=ResizeMethod.BILINEAR
        )

        # rotate
        image = tf.contrib.image.rotate(image, do_rotate)

        # flip
        image = tf.cond(
            do_flip > 0.5,
            lambda: tf.image.flip_left_right(image),
            lambda: image
        )
    else:
        h_aug_offset = 0
        w_aug_offset = 0

    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    offset_h = offset_h + tf.to_int32(tf.to_float(offset_h) * h_aug_offset)
    offset_w = offset_w + tf.to_int32(tf.to_float(offset_w) * w_aug_offset)

    image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, target_h, target_w)

    # shape defination
    image.set_shape([256, 256, 3])

    return image

def uvxyz_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0]), dtype=tf.float32):
    # load features
    height = tf.to_int32(features['uvxyz/height'])
    width = tf.to_int32(features['uvxyz/width'])

    uvxyz = tf.decode_raw(features['uvxyz'], dtype)
    uvxyz = tf.reshape(uvxyz, [height, width, -1])
    uvxyz_mask = tf.image.decode_jpeg(features['uvxyz/mask'], channels=1)
    
    # shape defination
    uvxyz.set_shape([256, 256, 3])
    return uvxyz


def heatmap_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0]), n_lms=16, flip_transformation=None):
    if flip_transformation is None:
        flip_transformation = list(range(n_lms))

    # load features
    n_landmarks = tf.to_int32(features['n_landmarks'])
    gt_lms = tf.decode_raw(features['gt'], tf.float32)
    if 'visible' in features:
        visible = tf.to_int32(tf.decode_raw(features['visible'], tf.int64))
    else:
        visible = tf.range(n_landmarks)
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])

    # formation
    gt_lms = tf.reshape(gt_lms, (n_landmarks, 2))
    gt_heatmap = utils.tf_lms_to_heatmap(
        gt_lms, image_height, image_width, n_landmarks, visible)
    gt_heatmap = tf.transpose(gt_heatmap, perm=[1, 2, 0])

    # augmentation
    if aug:
        do_flip, do_rotate, do_scale, h_aug_offset, w_aug_offset = tf.unstack(aug_args)

        # scale
        image_height = tf.to_int32(tf.to_float(image_height) * do_scale)
        image_width = tf.to_int32(tf.to_float(image_width) * do_scale)

        gt_heatmap = tf.image.resize_images(
            gt_heatmap,
            tf.stack([image_height, image_width]),
            method=ResizeMethod.BILINEAR)

        # rotate
        gt_heatmap = tf.contrib.image.rotate(gt_heatmap, do_rotate)

        # flip
        def flip_fn(gt_heatmap=gt_heatmap):
            gt_heatmap = tf.image.flip_left_right(gt_heatmap)

            flip_hm_list = []
            for idx in flip_transformation:
                flip_hm_list.append(gt_heatmap[:, :, idx])

            gt_heatmap = tf.stack(flip_hm_list, axis=2)

            return gt_heatmap

        gt_heatmap = tf.cond(
            do_flip > 0.5,
            flip_fn,
            lambda: gt_heatmap
        )
    else:
        h_aug_offset = 0
        w_aug_offset = 0

    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    offset_h = offset_h + tf.to_int32(tf.to_float(offset_h) * h_aug_offset)
    offset_w = offset_w + tf.to_int32(tf.to_float(offset_w) * w_aug_offset)

    gt_heatmap = tf.image.crop_to_bounding_box(
        gt_heatmap, offset_h, offset_w, target_h, target_w)

    # shape defination
    gt_heatmap.set_shape([256, 256, n_lms])

    return gt_heatmap


heatmap_resolver_pose_16 = functools.partial(
    heatmap_resolver,
    n_lms=16,
    flip_transformation=[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]
)


heatmap_resolver_face = functools.partial(
    heatmap_resolver,
    n_lms=68
)


def iuv_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0]),
                 n_parts=26, from_image=False, dtype=tf.int64):
    # load features
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])
    iuv_height = tf.to_int32(features['iuv_height'])
    iuv_width = tf.to_int32(features['iuv_height'])

    if from_image:
        iuv = tf.image.decode_jpeg(features['iuv'], channels=3)
    else:
        iuv = tf.decode_raw(features['iuv'], dtype)

    iuv = tf.reshape(iuv, [iuv_height, iuv_width, -1])
    iuv_mask = tf.to_int32(iuv[..., :1])
    uv = tf.to_float(iuv[..., 1:])

    # one hot mask
    iuv_one_hot = tf.contrib.layers.one_hot_encoding(tf.reshape(iuv_mask, [-1]), n_parts)
    iuv_one_hot = tf.reshape(iuv_one_hot, [iuv_height, iuv_width, n_parts])

    # normalised uv
    u = iuv_one_hot * uv[..., 0][..., None]
    v = iuv_one_hot * uv[..., 1][..., None]

    iuv = tf.concat([iuv_one_hot, u, v], 2)

    iuv = tf.concat([
        iuv[..., :1] - 1,
        iuv[..., 1:]], 2)

    # augmentation

    if aug:
        do_flip, do_rotate, do_scale, h_aug_offset, w_aug_offset = tf.unstack(aug_args)

        # scale
        image_height = tf.to_int32(tf.to_float(image_height) * do_scale)
        image_width = tf.to_int32(tf.to_float(image_width) * do_scale)

        iuv = tf.image.resize_images(
            iuv,
            tf.stack([image_height, image_width]),
            method=ResizeMethod.NEAREST_NEIGHBOR
        )

        # rotate
        iuv = tf.contrib.image.rotate(iuv, do_rotate)

        # flip
        iuv = tf.cond(
            do_flip > 0.5,
            lambda: tf.image.flip_left_right(iuv),
            lambda: iuv
        )
    else:
        h_aug_offset = 0
        w_aug_offset = 0

    iuv = tf.concat([
        iuv[..., :1] + 1,
        iuv[..., 1:]], 2)
    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    offset_h = offset_h + tf.to_int32(tf.to_float(offset_h) * h_aug_offset)
    offset_w = offset_w + tf.to_int32(tf.to_float(offset_w) * w_aug_offset)

    iuv = tf.image.crop_to_bounding_box(
        iuv, offset_h, offset_w, target_h, target_w)

    # shape defination
    iuv.set_shape([256, 256, n_parts * 3])

    return iuv


def image_file_resolver(content, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0])):
    image = tf.image.decode_jpeg(content)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_channels = tf.shape(image)[2]

    image = tf.cond(image_channels > 1,
                    lambda: image,
                    lambda: tf.image.grayscale_to_rgb(image))
    image = tf.to_float(image) / 255.

    # augmentation
    if aug:
        do_flip, do_rotate, do_scale, h_aug_offset, w_aug_offset = tf.unstack(aug_args)

        # scale
        image_height = tf.to_int32(tf.to_float(image_height) * do_scale)
        image_width = tf.to_int32(tf.to_float(image_width) * do_scale)

        image = tf.image.resize_images(
            image,
            tf.stack([image_height, image_width]),
            method=ResizeMethod.BILINEAR
        )

        # rotate
        image = tf.contrib.image.rotate(image, do_rotate)

        # flip
        image = tf.cond(
            do_flip > 0.5,
            lambda: tf.image.flip_left_right(image),
            lambda: image
        )
    else:
        h_aug_offset = 0
        w_aug_offset = 0

    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    offset_h = offset_h + tf.to_int32(tf.to_float(offset_h) * h_aug_offset)
    offset_w = offset_w + tf.to_int32(tf.to_float(offset_w) * w_aug_offset)

    image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, target_h, target_w)

    # shape defination
    image.set_shape([256, 256, 3])

    return image


def image_bbox_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0]), crop_size=321, final_size=256):
    # load features
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])

    n_landmarks = tf.to_int32(features['n_landmarks'])
    visible = tf.to_int32(tf.decode_raw(features['marked'], tf.int64))
    gt_lms = tf.decode_raw(features['rlms'], tf.float32)
    gt_lms = tf.reshape(gt_lms, [n_landmarks, 2])
    gt_lms_v = tf.gather(gt_lms, visible)

    # formation
    image = tf.reshape(image, (image_height, image_width, 3))
    image = tf.to_float(image) / 255.

    bbox = tf.concat([
        tf.reduce_min(gt_lms_v, axis=0),
        tf.reduce_max(gt_lms_v, axis=0)
    ], 0)
    bbox = tf.reshape(bbox, [2, 2])
    centre = tf.reduce_mean(bbox, axis=0)
    bbox = bbox - centre
    bbox = bbox * 1.5
    bbox = bbox + centre
    bbox = tf.reshape(bbox, [4])
    bbox = tf.where(bbox < 0., tf.zeros_like(bbox), bbox)
    bbox = tf.where(bbox > 1., tf.ones_like(bbox), bbox)

    gt_lms = gt_lms - bbox[:2]
    gt_lms = gt_lms / (bbox[2:] - bbox[:2])

    image = tf.image.crop_and_resize(image[None, ...], bbox[None, ...], [
                                     0], [crop_size, crop_size])[0]

    image_height = crop_size
    image_width = crop_size

    # augmentation
    if aug:
        do_flip, do_rotate, do_scale, h_aug_offset, w_aug_offset = tf.unstack(aug_args)

        # scale
        image_height = tf.to_int32(tf.to_float(image_height) * do_scale)
        image_width = tf.to_int32(tf.to_float(image_width) * do_scale)

        image = tf.image.resize_images(
            image,
            tf.stack([image_height, image_width]),
            method=ResizeMethod.BILINEAR
        )

        # rotate
        image = tf.contrib.image.rotate(image, do_rotate)

        # flip
        image = tf.cond(
            do_flip > 0.5,
            lambda: tf.image.flip_left_right(image),
            lambda: image
        )
    else:
        h_aug_offset = 0
        w_aug_offset = 0

    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    offset_h = offset_h + tf.to_int32(tf.to_float(offset_h) * h_aug_offset)
    offset_w = offset_w + tf.to_int32(tf.to_float(offset_w) * w_aug_offset)

    image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, target_h, target_w)

    # shape defination
    image.set_shape([256, 256, 3])

    return image


def heatmap_bbox_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0]), crop_size=321, final_size=256):
    # load features
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])

    n_landmarks = tf.to_int32(features['n_landmarks'])
    visible = tf.to_int32(tf.decode_raw(features['marked'], tf.int64))
    gt_lms = tf.decode_raw(features['rlms'], tf.float32)
    gt_lms = tf.reshape(gt_lms, [n_landmarks, 2])
    gt_lms_v = tf.gather(gt_lms, visible)

    # formation
    image = tf.reshape(image, (image_height, image_width, 3))
    image = tf.to_float(image) / 255.

    bbox = tf.concat([
        tf.reduce_min(gt_lms_v, axis=0),
        tf.reduce_max(gt_lms_v, axis=0)
    ], 0)
    bbox = tf.reshape(bbox, [2, 2])
    centre = tf.reduce_mean(bbox, axis=0)
    bbox = bbox - centre
    bbox = bbox * 1.5
    bbox = bbox + centre
    bbox = tf.reshape(bbox, [4])
    bbox = tf.where(bbox < 0., tf.zeros_like(bbox), bbox)
    bbox = tf.where(bbox > 1., tf.ones_like(bbox), bbox)

    gt_lms = gt_lms - bbox[:2]
    gt_lms = gt_lms / (bbox[2:] - bbox[:2])

    image = tf.image.crop_and_resize(image[None, ...], bbox[None, ...], [
                                     0], [crop_size, crop_size])[0]
    gt_heatmap = utils.tf_lms_to_heatmap(
        gt_lms * [crop_size, crop_size], crop_size, crop_size, n_landmarks, visible, sigma=7)
    gt_heatmap = tf.transpose(gt_heatmap, perm=[1, 2, 0])

    image_height = crop_size
    image_width = crop_size

    # augmentation
    if aug:
        do_flip, do_rotate, do_scale, h_aug_offset, w_aug_offset = tf.unstack(aug_args)

        # scale
        image_height = tf.to_int32(tf.to_float(image_height) * do_scale)
        image_width = tf.to_int32(tf.to_float(image_width) * do_scale)

        gt_heatmap = tf.image.resize_images(
            gt_heatmap,
            tf.stack([image_height, image_width]),
            method=ResizeMethod.BILINEAR)

        # rotate
        gt_heatmap = tf.contrib.image.rotate(gt_heatmap, do_rotate)

        # flip
        def flip_fn(gt_heatmap=gt_heatmap):
            gt_heatmap = tf.image.flip_left_right(gt_heatmap)

            flip_hm_list = []
            for idx in [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]:
                flip_hm_list.append(gt_heatmap[:, :, idx])

            gt_heatmap = tf.stack(flip_hm_list, axis=2)

            return gt_heatmap

        gt_heatmap = tf.cond(
            do_flip > 0.5,
            flip_fn,
            lambda: gt_heatmap
        )
    else:
        h_aug_offset = 0
        w_aug_offset = 0

    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    offset_h = offset_h + tf.to_int32(tf.to_float(offset_h) * h_aug_offset)
    offset_w = offset_w + tf.to_int32(tf.to_float(offset_w) * w_aug_offset)

    gt_heatmap = tf.image.crop_to_bounding_box(
        gt_heatmap, offset_h, offset_w, target_h, target_w)

    # shape defination
    gt_heatmap.set_shape([256, 256, 16])

    return gt_heatmap


def cyclegan_image_file_resolver(content, aug=False, aug_args=tf.constant([0, 0, 1])):
    image = tf.image.decode_jpeg(content)
    image_channels = tf.shape(image)[2]

    image = tf.cond(image_channels > 1,
                    lambda: image,
                    lambda: tf.image.grayscale_to_rgb(image))
    image = tf.to_float(image) / 255. * 2. - 1.

    # augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_images(image, [286, 286])
    image = tf.random_crop(image, [256, 256, 3])

    # shape defination
    image.set_shape([256, 256, 3])

    return image


def paired_image_file_resolver(content, aug=False, aug_args=tf.constant([0, 0, 1])):

    image = tf.image.decode_jpeg(content)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_channels = tf.shape(image)[2]

    image = tf.cond(image_channels > 1,
                    lambda: image,
                    lambda: tf.image.grayscale_to_rgb(image))

    image_channels = 3

    image = tf.reshape(
        tf.transpose(
            tf.reshape(
                image, [image_height, 2, image_width // 2, image_channels]
            ), [0, 2, 1, 3]
        ), [image_height, image_width // 2, image_channels * 2])

    image = tf.to_float(image) / 255. * 2. - 1.

    # augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_images(image, [286, 286])
    image = tf.random_crop(image, [256, 256, image_channels * 2])

    # shape defination
    image.set_shape([256, 256, 6])

    return image


def paired_seq_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0])):
    frames = features['frames'].values
    drawings = features['drawings'].values
    n_data = tf.shape(frames)[0]
    sliding_window = 3

    # formating
    frames = tf.to_float(tf.map_fn(tf.image.decode_jpeg,
                                   frames, dtype=tf.uint8)) / 255.
    drawings = 1 - \
        tf.to_float(tf.map_fn(tf.image.decode_jpeg,
                              drawings, dtype=tf.uint8)) / 255.
    paired_squence = tf.concat([frames, drawings], -1)
    image_height = tf.shape(frames)[1]
    image_width = tf.shape(frames)[2]

    # centre crop 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    paired_squence = tf.image.crop_to_bounding_box(
        paired_squence, offset_h, offset_w, target_h, target_w)

    # build sliding window

    range_indexes = tf.range(0, n_data - sliding_window + 1)
    sequences = tf.map_fn(
        lambda x: paired_squence[x:x+sliding_window], range_indexes, dtype=tf.float32)
    sequences = sequences * 2 - 1

    sequences.set_shape([None, sliding_window, None, None, 6])

    return sequences


def paired_masked_seq_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1, 0, 0])):
    frames = features['frames'].values
    drawings = features['drawings'].values
    masks = features['masks']
    n_data = tf.shape(frames)[0]
    sliding_window = 3

    # formating
    frames = tf.to_float(tf.map_fn(tf.image.decode_jpeg,
                                   frames, dtype=tf.uint8)) / 255.
    drawings = 1 - \
        tf.to_float(tf.map_fn(tf.image.decode_jpeg,
                              drawings, dtype=tf.uint8)) / 255.
    masks = tf.reshape(tf.to_float(tf.decode_raw(
        masks, tf.uint8)), [n_data, 384, 384, 3])
    masks = (masks + 1.) / 2.

    image_height = tf.shape(frames)[1]
    image_width = tf.shape(frames)[2]

    # merge by channels
    paired_squence = tf.concat([frames, drawings, masks], -1)

    # centre crop 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    paired_squence = tf.image.crop_to_bounding_box(
        paired_squence, offset_h, offset_w, target_h, target_w)

    # build sliding window

    range_indexes = tf.range(0, n_data - sliding_window + 1)
    sequences = tf.map_fn(
        lambda x: paired_squence[x:x+sliding_window], range_indexes, dtype=tf.float32)
    sequences = sequences * 2 - 1

    sequences.set_shape([None, sliding_window, None, None, 9])

    return sequences


def decode_jpeg(feature, *args, **kargs):
    return tf.image.decode_jpeg(feature['image'])


def decode_mask(feature, *args, **kargs):
    return tf.image.decode_png(feature['mask'])


ResolveMaskedImage = {
    'inputs': decode_jpeg,
    'masks': decode_mask
}

ResolveMaskedPairedSeq = {
    'inputs': paired_masked_seq_resolver,
    'dummy': dummy_seq_resolver
}


ResolvePairedSeq = {
    'inputs': paired_seq_resolver,
    'dummy': dummy_seq_resolver
}


ResolverPairedImage = {
    'inputs': paired_image_file_resolver,
    'dummy': dummy_resolver
}


ResolverImage = {
    'inputs': cyclegan_image_file_resolver,
    'dummy': dummy_resolver
}

ResolverHMPose = {
    'inputs': image_resolver,
    'heatmap': heatmap_resolver_pose_16,
}

ResolverBBoxPose = {
    'inputs': image_bbox_resolver,
    'heatmap': heatmap_bbox_resolver
}

ResolverIUVHM = {
    'inputs': image_resolver,
    'heatmap': heatmap_resolver_pose_16,
    'iuv': iuv_resolver
}

ResolverIUV = {
    'inputs': image_resolver,
    'iuv': functools.partial(iuv_resolver, from_image=False, dtype=tf.uint8)
}

ResolverHMFace = {
    'inputs': image_resolver,
    'heatmap': heatmap_resolver_face,
}

ResolverIUVFace = {
    'inputs': image_resolver,
    'heatmap': heatmap_resolver_face,
    'uv': iuv_resolver
}
