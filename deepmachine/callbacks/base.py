import tensorflow as tf

import functools

slim = tf.contrib.slim
K = tf.keras.backend

from .. import utils
from tensorflow.python.summary import summary as tf_summary


def summary_iuv(data, shape=None, name='iuvs'):

    batch_size, height, width, channels = shape or data.get_shape().as_list()
    iuv_rgb = utils.tf_iuv_rgb(data, n_feature=channels//3)

    # iuv summary
    tf.summary.image(
        name,
        iuv_rgb,
        max_outputs=3)


def summary_landmarks(data, shape=None, name='ladnmarks'):
    batch_size, height, width, channels = shape or data.get_shape().as_list()

    # landmarks summary
    tf.summary.image(
        name,
        utils.tf_n_channel_rgb(data, channels),
        max_outputs=3)


def summary_batch(data, shape=None, name='batch', col_size=4):
    batch_size, height, width, channels = shape or data.get_shape().as_list()

    tf.summary.image(
        name,
        tf.map_fn(
            functools.partial(utils.tf_image_batch_to_grid, col_size=col_size),
            tf.reshape(tf.transpose(
                data, [0, 3, 1, 2]), [batch_size, -1, height, width, 1])
        ),
        max_outputs=3)


def summary_image_batch(data, shape=None, name='image_batch', col_size=4):

    batch_size, height, width, channels = shape or data.get_shape().as_list()

    tf.summary.image(
        name,
        tf.map_fn(
            functools.partial(utils.tf_image_batch_to_grid, col_size=col_size),
            tf.transpose(tf.reshape(tf.transpose(
                data, [0, 3, 1, 2]), [batch_size, -1, 3, height, width]), [0,1,3,4,2])
        ),
        max_outputs=3)


class TBSummary(tf.keras.callbacks.TensorBoard):

    def set_model(self, model):

        # add learning rate summary
        optimizer = model.optimizer
        tf.summary.scalar('learning_rate', optimizer.lr)

        # add image summary
        for index, tf_inputs in enumerate(model.inputs):
            name = 'default_summary/inputs_%02d' % index
            # normalise images
            tf_inputs_shape = tf_inputs.shape.as_list()
            tf_inputs_shape[0] = tf_inputs_shape[0] or self.batch_size

            if tf_inputs_shape[-1] in [1, 3, 4]:
                tf.summary.image(name, tf_inputs)
            else:
                summary_batch(tf_inputs, shape=tf_inputs_shape, name=name)

        for index, (tf_targets, tf_outputs) in enumerate(zip(model.targets, model.outputs)):
            name = 'default_summary/{}_%02d' % index
            tf_shape = tf_outputs.shape.as_list()
            tf_shape[0] = tf_shape[0] or self.batch_size

            # normalise images

            if tf_shape[-1] in [1, 3, 4]:
                tf.summary.image(name.format('targets'), tf_targets)
                tf.summary.image(name.format('outputs'), tf_outputs)
            else:
                summary_batch(tf_targets, shape=tf_shape,
                              name=name.format('targets'))
                summary_batch(tf_outputs, shape=tf_shape,
                              name=name.format('outputs'))

        if hasattr(model, 'summaries') and model.summaries and len(model.summaries) == 2:
            summaries = []
            if type(model.summaries[0]) is not list:
                summaries += [model.summaries[0]]
            else:
                summaries += model.summaries[0]

            if type(model.summaries[1]) is not list:
                summaries += [model.summaries[1] for _ in range(2)]
            else:
                summaries += model.summaries[1] + model.summaries[1]

            tensors = model.inputs + model.outputs + model.targets
            assert len(tensors) == len(summaries)
            for summary_fn, tensor in zip(summaries, tensors):
                if callable(summary_fn):
                    summary_fn(
                        tensor, name='{}/{}'.format('user_summaries', tensor.name.replace(':0','')))

        super().set_model(model)

    def on_batch_end(self, batch, logs=None):
        # rewrite on_epoch_end to support tensor input visualisation
        logs = logs or {}

        logs_print = logs.copy()

        logs_print.update({
            'current_epoch': self.model.epoch,
            'current_batch': batch + 1
        })

        self.model.progress_bar.update(
            self.model.epoch * self.model.steps_per_epoch + batch + 1,
            values=logs_print.items()
        )

        super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # rewrite on_epoch_end to support tensor input visualisation
        logs = logs or {}

        logs_print = logs.copy()

        logs_print.update({
            'current_epoch': epoch + 1,
            'current_batch': 0
        })

        self.model.epoch = epoch + 1

        self.model.progress_bar.update(
            self.model.epoch * self.model.steps_per_epoch,
            values=logs_print.items()
        )

        if self.histogram_freq:
            if not self.validation_data:
                result = self.sess.run([self.merged])
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

            else:
                if epoch % self.histogram_freq == 0:

                    val_data = self.validation_data
                    tensors = (
                        self.model.inputs + self.model.targets + self.model.sample_weights)

                    if self.model.uses_learning_phase:
                        tensors += [K.learning_phase()]

                    assert len(val_data) == len(tensors)
                    val_size = val_data[0].shape[0]
                    if not isinstance(val_size, int):
                        val_size = self.batch_size

                    i = 0
                    while i < val_size:
                        step = min(self.batch_size, val_size - i)
                        batch_val = []
                        batch_val.append(val_data[0][i:i + step]
                                         if val_data[0] is not None else None)
                        batch_val.append(val_data[1][i:i + step]
                                         if val_data[1] is not None else None)
                        batch_val.append(val_data[2][i:i + step]
                                         if val_data[2] is not None else None)
                        if self.model.uses_learning_phase:
                            # do not slice the learning phase
                            batch_val = [x[i:i + step] if x is not None else None
                                         for x in val_data[:-1]]
                            batch_val.append(val_data[-1])
                        else:
                            batch_val = [x[i:i + step] if x is not None else None
                                         for x in val_data]
                        feed_dict = {}
                        for key, val in zip(tensors, batch_val):
                            if val is not None:
                                if isinstance(val, tf.Tensor):
                                    val = self.sess.run(val)
                                feed_dict[key] = val
                        result = self.sess.run(
                            [self.merged], feed_dict=feed_dict)
                        summary_str = result[0]
                        self.writer.add_summary(summary_str, epoch)
                        i += self.batch_size

            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf_summary.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item() if hasattr(value, 'item') else value
                summary_value.tag = name
                self.writer.add_summary(summary, epoch)
            
            self.writer.flush()

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.model.epoch = 0
        logs_print = logs.copy()

        logs_print.update({
            'current_epoch': self.model.epoch,
            'current_batch': 0
        })

        self.model.progress_bar.update(
            self.model.epoch * self.model.steps_per_epoch,
            values=logs_print.items()
        )

        self.writer.reopen()

        super().on_train_begin(logs)