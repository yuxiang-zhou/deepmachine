import tensorflow as tf
import functools
import numpy as np

from pathlib import Path

from ... import utils

ResizeMethod = tf.image.ResizeMethod


class Provider():
    # flip, rotate, scale, crop
    def _random_augmentation(self):
        return tf.concat(
            [
                tf.random_uniform([1]),
                (tf.random_uniform([1]) * 60. - 30.) * np.pi / 180.,
                tf.random_uniform([1]) * 0.5 + 0.75,
                (tf.random_uniform([1]) * 2 - 1) * 0.2,
                (tf.random_uniform([1]) * 2 - 1) * 0.2
            ], 0)

    def get(self, *keys):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError


class TFRecordProvider(Provider):
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
        self._count = None

    def register_key_resolver(self, key, resolver):
        self._key_resolver[key] = resolver

    def get(self, *keys):
        data = self._get_data_protobuff(self._filename, *keys)
        images = data[0]
        names = data[1:]
        tensors = [images]

        for name in names:
            tensors.append(name)

        inputs_batch = tf.train.shuffle_batch(
            tensors, self._batch_size, 100, 20, 4)

        retval = {}
        for k, b in zip(keys, inputs_batch):
            retval[k] = b

        return retval

    def size(self):
        if self._count is None:
            self._count = 0
            for fn in str(self._filename).split(','):
                w_it = utils.tf_records_iterator(fn)
                self._count += len([1 for _ in w_it])

        return self._count

    def _get_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features=self._features
        )
        return features

    # Data from protobuff
    def _get_data_protobuff(self, filename, *keys):
        filenames = str(filename).split(',')
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=None)

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


class TFSeqRecordProvider(TFRecordProvider):
    def get(self, *keys):
        data = self._get_data_protobuff(self._filename, *keys)
        images = data[0]
        names = data[1:]
        tensors = [images]

        for name in names:
            tensors.append(name)

        inputs_batch = tf.train.shuffle_batch(
            tensors, self._batch_size, 100, 20, 4, enqueue_many=True)

        retval = {}
        for k, b in zip(keys, inputs_batch):
            retval[k] = b

        return retval


class TFRecordNoFlipProvider(TFRecordProvider):
    def _random_augmentation(self):
        return super()._random_augmentation() - tf.constant([1., 0., 0., 0., 0.])


class TFRecordBBoxProvider(TFRecordProvider):
    def _random_augmentation(self):
        return tf.concat(
            [
                tf.random_uniform([1]),
                (tf.random_uniform([1]) * 30. - 15.) * np.pi / 180.,
                tf.random_uniform([1]) * 0.4 + 0.8,
                (tf.random_uniform([1]) * 2 - 1) * 0.2,
                (tf.random_uniform([1]) * 2 - 1) * 0.2
            ], 0)


class TFDirectoryProvider(Provider):
    def __init__(self,
                 dirpath,
                 batch_size=1,
                 augmentation=False,
                 ext='.jpg',
                 no_processes=4,
                 resolvers={}):

        self._dirpath = dirpath
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._ext = ext
        self._key_resolver = resolvers
        self._count = None
        self.no_processes = no_processes

    def size(self):
        if self._count is None:
            self._count = 0
            for path in self._dirpath.split(','):
                self._count += len(list(Path(path).glob('*.jpg')))

        return self._count

    def get(self, *keys):

        filelist = []
        for path in self._dirpath.split(','):
            filelist += list(map(str, Path(path).glob('*.jpg')))

        tf_filelist = tf.convert_to_tensor(filelist)
        self._count = n_items = len(filelist)
        print('Number of items: %d' % n_items)

        producer = tf.train.string_input_producer(
            tf_filelist, capacity=n_items)

        queue = None
        enqueue_ops = []
        for _ in range(self.no_processes):
            tf_filepath = producer.dequeue()
            tf_file_contents = tf.read_file(tf_filepath)

            augmentation_args = self._random_augmentation()

            tensors = []
            for key in keys:
                fn = functools.partial(
                    self._key_resolver[key],
                    aug=self._augmentation,
                    aug_args=augmentation_args
                )
                tensors.append(fn(tf_file_contents))

            shapes = [x.get_shape() for x in tensors]

            if queue is None:
                dtypes = [x.dtype for x in tensors]
                queue = tf.FIFOQueue(
                    capacity=200,
                    dtypes=dtypes, name='fifoqueue')

            enqueue_ops.append(queue.enqueue(tensors))

        qr = tf.train.QueueRunner(queue, enqueue_ops)
        tf.train.add_queue_runner(qr)

        tensors_dequeue = queue.dequeue()

        for t, s in zip(tensors_dequeue, shapes):
            t.set_shape(s)

        inputs_batch = tf.train.batch(
            tensors_dequeue, batch_size=self._batch_size, enqueue_many=False,
            capacity=4 * self.no_processes * self._batch_size,
            allow_smaller_final_batch=True, dynamic_pad=True,)

        retval = {}
        for k, b in zip(keys, inputs_batch):
            retval[k] = b

        return retval


class DatasetMixer(Provider):
    def __init__(self,
                 providers,
                 batch_size=1):
        self.providers = providers
        self.batch_size = batch_size
        self._count = None

    def size(self):
        if self._count is None:
            self._count = 0
            self._count += sum([p.size() for p in self.providers])

        return self._count

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
            num_threads=1,
            enqueue_many=False,
            dynamic_pad=True,
            capacity=200)

        ret_dict = {k: v for k, v in zip(keys, ret_val)}

        return ret_dict


class DatasetQueue(Provider):
    def __init__(self,
                 provider,
                 n_proccess=4,
                 batch_size=1):
        self.provider = provider
        self.batch_size = batch_size
        self.n_proccess = n_proccess
        self._count = None

    def size(self):
        if self._count is None:
            self._count = self.provider.size()

        return self._count

    def get(self, *keys):

        tensor_dict = self.provider.get(*keys)
        tensors = [tensor_dict[k] for k in keys]
        dtypes = [x.dtype for x in tensors]
        shapes = [x.get_shape() for x in tensors]
        queue = tf.FIFOQueue(
            capacity=self.batch_size * 10 * self.n_proccess,
            dtypes=dtypes, name='fifoqueue')

        qr = tf.train.QueueRunner(
            queue, [queue.enqueue_many(tensors)] * self.n_proccess)
        tf.train.add_queue_runner(qr)

        tensors_dequeue = queue.dequeue()

        for t, s in zip(tensors_dequeue, shapes):
            t.set_shape(s[1:])

        ret_val = tf.train.batch(
            tensors_dequeue,
            self.batch_size,
            num_threads=1,
            enqueue_many=False,
            dynamic_pad=True,
            capacity=200)

        ret_dict = {k: v for k, v in zip(keys, ret_val)}

        return ret_dict


class DatasetPairer(Provider):
    def __init__(self,
                 providers,
                 batch_size=1):
        self.providers = providers
        self.batch_size = batch_size
        self._count = None

    def size(self):
        if self._count is None:
            self._count = 0
            self._count = min([p.size() for p in self.providers])

        return self._count

    def get(self, *keys):

        tensors_all = []
        n_providers = len(self.providers)
        n_keys = len(keys)

        for p in self.providers:
            tensor_dict = p.get(*keys)
            tensors = [tensor_dict[k] for k in keys]
            tensors_all += tensors

        merged_batch = []
        for tensors_to_merge in [tensors_all[i::n_keys] for i in range(n_providers)]:
            tensor = tf.concat(tensors_to_merge, -1)
            merged_batch.append(tensor)

        ret_val = tf.train.batch(
            merged_batch,
            self.batch_size,
            num_threads=1,
            enqueue_many=True,
            dynamic_pad=True,
            capacity=200)

        ret_dict = {k: v for k, v in zip(keys, ret_val)}

        return ret_dict
