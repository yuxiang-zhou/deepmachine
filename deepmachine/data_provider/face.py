import numpy as np
import tensorflow as tf

from pathlib import Path

from ..flags import FLAGS

def generate_occlusion(image, uv):
    def wrap(image, uv):
        for _ in range(int(np.random.rand() * 5)):
            min_x = int(np.random.uniform(0, 200))
            max_x = int(min(199, min_x + np.random.uniform(10, 50)))

            min_y = int(np.random.uniform(0, 200))
            max_y = int(min(199, min_y + np.random.uniform(10, 50)))

            image[min_y:max_y, min_x:max_x] = np.random.rand(
                max_y - min_y, max_x - min_x, 1) * 255
            # uv[min_y:max_y, min_x:max_x] = 0
        return image, uv

    return tf.py_func(wrap, [image, uv], [tf.float32, tf.float32])


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    # RGB -> BGR
    image = tf.reverse(image, [2])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image


class Dataset:
    uv_channels = 3

    def __init__(self, names, batch_size=8, is_training=False):
        self.batch_size = batch_size
        self.root = Path('/vol/atlas/databases/tf_records/')
        self.tfrecord_names = names
        self.is_training = is_training

    def get(self):
        paths = [str(self.root / x) for x in self.tfrecord_names]

        filename_queue = tf.train.string_input_producer(paths)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'uv': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
            })

        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.to_float(image)

        height, width = 600, 600

        image.set_shape((height, width, 3))

        uv = tf.decode_raw(features['uv'], tf.float32)

        uv.set_shape((self.uv_channels * width * height))
        uv = tf.reshape(uv, (height, width, self.uv_channels))

        if self.is_training:
            begin = tf.random_uniform((), 0, 20, tf.int32)
            size = tf.reduce_min(
                [600 - begin, tf.random_uniform((), 555, 600, tf.int32)])

            image = tf.slice(image, [begin, begin, 0], [size, size, -1])
            uv = tf.slice(uv, [begin, begin, 0], [size, size, -1])

            image = tf.image.resize_images(image, (200, 200), method=0)
            uv = tf.image.resize_images(uv, (200, 200), method=1)[..., :2]

            # image, uv = generate_occlusion(image, uv)
            image.set_shape((200, 200, 3))
            uv.set_shape((200, 200, 2))

            image = distort_color(image / 255.) * 255.

        image = caffe_preprocess(image)
        mask = tf.to_float(tf.reduce_mean(uv, 2) >= 0)[..., None]

        return tf.train.shuffle_batch(
            [image, uv, mask],
            self.batch_size,
            capacity=1000,
            num_threads=2,
            min_after_dequeue=200)


class SyntheticPoseDataset(Dataset):
    uv_channels = 2

    def __init__(self, **kwargs):
        names = ['synthetic_pose_densereg_600x600.tfrecords']
        super().__init__(names, **kwargs)

    def num_samples(self):
        return 10000


class SyntheticDataset(Dataset):
    def __init__(self, **kwargs):
        names = ['synthetic_densereg_600x600.tfrecords']
        super().__init__(names, **kwargs)

    def num_samples(self):
        return 14193 * 3


class Dataset300W(Dataset):
    def __init__(self, **kwargs):
        names = ['300w_densereg_600x600.tfrecords',
                 'helen_densereg_600x600.tfrecords']
        super().__init__(names, **kwargs)

    def num_samples(self):
        return 600 + 2000


class HelenDataset(Dataset):
    def __init__(self, **kwargs):
        names = ['helen_densereg_600x600.tfrecords']
        super().__init__(names, **kwargs)

    def num_samples(self):
        return 2000


class MenpoDataset(Dataset):
    def __init__(self, **kwargs):
        names = ['menpo_trainset_densereg_600x600.tfrecords']
        super().__init__(names, **kwargs)

    def num_samples(self):
        return 8954


class FaceDatasetMixer():
    def __init__(self, names, densities=None, batch_size=1, is_training=False):
        self.providers = []
        self.batch_size = batch_size

        if densities is None:
            densities = [1] * len(names)

        for name, bs in zip(names, densities):
            provider = globals()[name](batch_size=bs, is_training=is_training)
            self.providers.append(provider)

    def get(self, **kargs):
        queue = None
        enqueue_ops = []
        for p in self.providers:
            tensors = p.get(**kargs)

            shapes = [x.get_shape() for x in tensors]

            if queue is None:
                dtypes = [x.dtype for x in tensors]
                queue = tf.FIFOQueue(
                    capacity=1000,
                    dtypes=dtypes, name='fifoqueue')

            enqueue_ops.append(queue.enqueue_many(tensors))

        qr = tf.train.QueueRunner(queue, enqueue_ops)
        tf.train.add_queue_runner(qr)

        tensors = queue.dequeue()

        for t, s in zip(tensors, shapes):
            t.set_shape(s[1:])

        train_batch = tf.train.batch(
            tensors,
            self.batch_size,
            num_threads=2,
            enqueue_many=False,
            dynamic_pad=True,
            capacity=200)

        return {
            'inputs': train_batch[0],
            'uv': train_batch[1],
            'mask': train_batch[2],
        }


DenseFaceProvider = FaceDatasetMixer(
    ['SyntheticDataset', 'MenpoDataset', 'HelenDataset'],
    batch_size=FLAGS.batch_size,
    densities=(1, 2, 1, 1),
    is_training=True)
