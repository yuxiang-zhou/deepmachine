import tensorflow as tf
import functools
import numpy as np

from .. import utils

slim = tf.contrib.slim
ResizeMethod = tf.image.ResizeMethod


class TFRecordProvider(object):
    def __init__(self,
                 filename,
                 features,
                 batch_size=1,
                 augmentation=True,
                 resolvers={}):
        self._filename = filename
        self._features = features
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._key_resolver = resolvers

    def register_key_resolver(self, key, resolver):
        self._key_resolver[key] = resolver

    def get(self, *keys):
        images, *names = self._get_data_protobuff(self._filename, *keys)
        tensors = [images]

        for name in names:
            tensors.append(name)

        inputs_batch = tf.train.shuffle_batch(
            tensors, self._batch_size, 100, 20, 4)

        retval = {}
        for k, b in zip(keys, inputs_batch):
            retval[k] = b

        return retval

    # flip, rotate, scale
    def _random_augmentation(self):
        return tf.concat([tf.random_uniform([1]),
                          (tf.random_uniform([1]) * 60. - 30.) * np.pi / 180.,
                          tf.random_uniform([1]) * 0.5 + 0.75], 0)

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features=self._features
        )
        return features

    # Data from protobuff
    def _get_data_protobuff(self, filename, *keys):
        filename = str(filename).split(',')
        filename_queue = tf.train.string_input_producer(
            filename, num_epochs=None)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = self._get_features(serialized_example)
        augmentation_args = self._random_augmentation()

        data = []
        for key in keys:
            fn = functools.partial(
                self._key_resolver[key],
                aug=self._augmentation,
                aug_args=augmentation_args
            )
            data.append(fn(features))

        return data


class DatasetMixer():
    def __init__(self,
                 providers,
                 batch_size=1):
        self.providers = providers
        self.batch_size = batch_size


    def get(self, *keys):
        queue = None
        enqueue_ops = []
        for p in self.providers:
            tensor_dict = p.get(*keys)
            tensors = [tensor_dict[k] for k in keys]
            shapes = [x.get_shape() for x in tensors]

            if queue is None:
                dtypes = [x.dtype for x in tensors]
                queue = tf.FIFOQueue(
                    capacity=200,
                    dtypes=dtypes, name='fifoqueue')

            enqueue_ops.append(queue.enqueue_many(tensors))

        qr = tf.train.QueueRunner(queue, enqueue_ops)
        tf.train.add_queue_runner(qr)

        tensors_dequeue = queue.dequeue()

        for t, s in zip(tensors_dequeue, shapes):
            t.set_shape(s[1:])

        ret_val = tf.train.batch(
            tensors_dequeue,
            self.batch_size,
            num_threads=2,
            enqueue_many=False,
            dynamic_pad=True,
            capacity=200)

        ret_dict = {k:v for k,v in zip(keys, ret_val)}

        return ret_dict


def image_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1])):
    # load features
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])

    # formation
    image = tf.reshape(image, (image_height, image_width, 3))
    image = tf.to_float(image) / 255.

    # augmentation
    if aug:
        do_flip, do_rotate, do_scale = tf.unstack(aug_args)

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

    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, target_h, target_w)

    # shape defination
    image.set_shape([None, None, 3])

    return image


def heatmap_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1])):
    # load features
    n_landmarks = tf.to_int32(features['n_landmarks'])
    gt_lms = tf.decode_raw(features['gt'], tf.float32)
    visible = tf.to_int32(tf.decode_raw(features['visible'], tf.int64))
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])

    # formation
    gt_lms = tf.reshape(gt_lms, (n_landmarks, 2))
    gt_heatmap = utils.tf_lms_to_heatmap(
        gt_lms, image_height, image_width, n_landmarks, visible)
    gt_heatmap = tf.transpose(gt_heatmap, perm=[1, 2, 0])

    # augmentation
    if aug:
        do_flip, do_rotate, do_scale = tf.unstack(aug_args)

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
        def flip(gt_heatmap=gt_heatmap):
            idx = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]
            gt_heatmap = tf.transpose(
                tf.gather(tf.transpose(gt_heatmap, [2, 0, 1]), idx),
                [1, 2, 0]
            )

            return gt_heatmap

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

    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    gt_heatmap = tf.image.crop_to_bounding_box(
        gt_heatmap, offset_h, offset_w, target_h, target_w)

    # shape defination
    gt_heatmap.set_shape([None, None, 16])

    return gt_heatmap


def iuv_resolver(features, aug=False, aug_args=tf.constant([0, 0, 1]), n_parts=26):
    # load features
    image_height = tf.to_int32(features['height'])
    image_width = tf.to_int32(features['width'])
    iuv = tf.to_int32(tf.decode_raw(features['iuv'], tf.int64))
    iuv_height = tf.to_int32(features['iuv_height'])
    iuv_width = tf.to_int32(features['iuv_height'])

    # formation
    iuv = tf.to_float(iuv)
    iuv = tf.reshape(iuv, (iuv_height, iuv_width, 3))

    # one hot mask
    iuv_mask = iuv[..., 0]

    iuv_one_hot = slim.one_hot_encoding(
        tf.to_int32(tf.reshape(iuv_mask, [-1])),
        n_parts)
    iuv_one_hot = tf.reshape(iuv_one_hot, [iuv_height, iuv_width, n_parts])

    # normalised uv
    uv = iuv[..., 1:] / 255.
    u = iuv_one_hot * uv[..., 0][..., None]
    v = iuv_one_hot * uv[..., 1][..., None]

    iuv = tf.concat([iuv_one_hot, u, v], 2)

    iuv = tf.concat([
        iuv[..., :1] - 1,
        iuv[..., 1:]], 2)

    # pad iuv
    pad_h = (image_height - iuv_height) // 2
    pad_w = (image_width - iuv_width) // 2
    iuv = tf.pad(iuv, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]])

    # augmentation

    if aug:
        do_flip, do_rotate, do_scale = tf.unstack(aug_args)

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

    iuv = tf.concat([
        iuv[..., :1] + 1,
        iuv[..., 1:]], 2)
    # crop to 256 * 256
    target_h = tf.to_int32(256)
    target_w = tf.to_int32(256)
    offset_h = tf.to_int32((image_height - target_h) / 2)
    offset_w = tf.to_int32((image_width - target_w) / 2)

    iuv = tf.image.crop_to_bounding_box(
        iuv, offset_h, offset_w, target_h, target_w)

    # shape defination
    iuv.set_shape([None, None, n_parts * 3])

    return iuv
