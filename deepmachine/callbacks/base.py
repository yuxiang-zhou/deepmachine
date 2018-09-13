import tensorflow as tf
import keras
import numpy as np
import functools
import collections
import matplotlib.pyplot as plt
from keras import backend as K
from io import BytesIO

slim = tf.contrib.slim

from .. import utils


def summary_iuv(data, shape=None, name='iuvs'):

    _, _, _, channels = shape or data.get_shape().as_list()
    iuv_rgb = utils.tf_iuv_rgb(data, n_feature=channels//3)

    # iuv summary
    tf.summary.image(
        name,
        iuv_rgb,
        max_outputs=3)


def summary_landmarks(data, shape=None, name='ladnmarks'):
    _, _, _, channels = shape or data.get_shape().as_list()

    # landmarks summary
    tf.summary.image(
        name,
        utils.tf_n_channel_rgb(data, channels),
        max_outputs=3)


def summary_batch(data, shape=None, name='batch', col_size=4):
    batch_size, height, width, _ = shape or data.get_shape().as_list()

    tf.summary.image(
        name,
        tf.map_fn(
            functools.partial(utils.tf_image_batch_to_grid, col_size=col_size),
            tf.reshape(tf.transpose(
                data, [0, 3, 1, 2]), [batch_size, -1, height, width, 1])
        ),
        max_outputs=3)


def summary_image_batch(data, shape=None, name='image_batch', col_size=4):

    batch_size, height, width, _ = shape or data.get_shape().as_list()

    tf.summary.image(
        name,
        tf.map_fn(
            functools.partial(utils.tf_image_batch_to_grid, col_size=col_size),
            tf.transpose(tf.reshape(tf.transpose(
                data, [0, 3, 1, 2]), [batch_size, -1, 3, height, width]), [0, 1, 3, 4, 2])
        ),
        max_outputs=3)


class Monitor(keras.callbacks.Callback):

    def __init__(self, logdir=None, models=None, restore=True, *args, **kwargs):
        self.logdir = logdir
        self.writer = tf.summary.FileWriter(logdir,  K.get_session().graph) if logdir else None
        self.models = models
        self.restore = restore
        self.history = {}
        self.valid_history = {}

    def _standarize_images(self, images):
        if images.min() < 0:
            images = (images.clip(-1, 1) + 1) / 2.

        if images.max() > 1:
            images = images.clip(0, 1)

        if images.shape[-1] not in [1, 3, 4]:
            images = np.array(list(map(utils.channels_to_grid, images)))

        return images

    def log_images(self, tag, images, step, max_images=4):
        """Logs a list of images."""
        images = self._standarize_images(images)
        im_summaries = []
        for nr, img in enumerate(images[:max_images]):
            # Write the image to a string
            s = BytesIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)


    def on_valid_batch(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.valid_history.setdefault(k, []).append(v)
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_begin(self, epoch, logs = None):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = utils.Summary(logs)

        # save smmaries to logdir
        if self.writer is not None:
            
            # save traing summaries
            for name, value in self.history.items():
                if type(value) is not str:
                    self.log_scalar('train/' + name, np.mean(value), epoch)

            # save validation summaries
            for name, value in self.valid_history.items():
                if type(value) is not str:
                    self.log_scalar('valid/' + name, np.mean(value), epoch)

            # save visualisations
            for name, value in logs.images.items():
                self.log_images('train/' + name, value, epoch)


            self.writer.flush()

        # save weights
        for m in self.models:
            if m:
                m.save('{}/{}-weights.{:05d}.hdf5'.format(
                    self.logdir,
                    m.name, epoch), include_optimizer=False)

        super().on_epoch_end(epoch, logs=logs)

    def on_train_begin(self, logs=None):
        if not isinstance(self.models, collections.Iterable):
            self.models = [self.models]

        # print summaries
        for m in self.models:
            if m:
                m.summary()

        # restore weights
        init_epoch = utils.max_epoch(self.logdir)
        if self.restore and init_epoch > 0:
            print('Restoring Previous Checkpoints with epoch: {}...'.format(init_epoch))
            for m in self.models:
                if m:
                    m.load_weights('{}/{}-weights.{:05d}.hdf5'.format(
                        self.logdir,
                        m.name, init_epoch))

        if self.writer is not None:
            self.writer.reopen()

        super().on_train_begin(logs)


class BatchHistory(keras.callbacks.Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        self.history = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            if len(self.history[k]) > 10:
                self.history[k].pop(0)
