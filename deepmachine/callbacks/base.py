import tensorflow as tf
import functools
slim = tf.contrib.slim
from .. import utils


def summary_input(data_eps, network_eps, is_training=True):
    inputs = data_eps['inputs']

    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[3]

    tf.summary.image(
        'images/batch',
        tf.map_fn(
            functools.partial(utils.tf_image_batch_to_grid, col_size=1),
            tf.transpose(tf.reshape(
                inputs, [batch_size, height, width, -1, 3]), [0, 3, 1, 2, 4])
        ),
        max_outputs=3)


def summary_input_LSTM(data_eps, network_eps, is_training=True, n_channels=3):
    input_seqs = data_eps['inputs']

    # inputs
    targets = data_eps['inputs'][:, 0, :, :, :n_channels]
    inputs = data_eps['inputs'][:, 0, :, :, n_channels:n_channels*2]
    masks = data_eps['inputs'][:, 0, :, :, n_channels*2:n_channels*3]
    targets *= masks

    tf.summary.image(
        'images/input_pair',
        tf.concat([inputs, targets], -2),
        max_outputs=3)


def summary_total_loss(data_eps, network_eps, is_training=True):
    if is_training:
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total_loss', total_loss)


def summary_uv(data_eps, network_eps, is_training=True):
    uvs = data_eps['uv']
    pred_uv, _ = network_eps

    tf.summary.image('gt/uvs/h', uvs[..., :1])
    tf.summary.image('gt/uvs/v', uvs[..., 1:])

    tf.summary.image(
        'predictions/cascade-regression',
        tf.map_fn(
            utils.tf_image_batch_to_grid,
            tf.transpose(pred_uv, [0, 3, 1, 2])[..., None]
        ),
        max_outputs=3)


def summary_iuv(data_eps, network_eps, is_training=True, n_feature=26):

    iuv_gt = data_eps['iuv']
    _, states = network_eps
    iuv_pred = states['iuv']

    iuv_gt_rgb = utils.tf_iuv_rgb(iuv_gt, n_feature=n_feature)
    iuv_pred_rgb = utils.tf_iuv_rgb(iuv_pred, n_feature=n_feature)

    # iuv summary
    tf.summary.image(
        'predictions/iuv',
        iuv_pred_rgb,
        max_outputs=3)

    tf.summary.image(
        'gt/iuv',
        iuv_gt_rgb,
        max_outputs=3)


def summary_landmarks(data_eps, network_eps, is_training=True, n_channel=16):
    gt_heatmap = data_eps['heatmap']
    predictions, _ = network_eps

    # landmarks summary
    tf.summary.image(
        'predictions/landmark-regression',
        utils.tf_n_channel_rgb(predictions, n_channel),
        max_outputs=3)

    tf.summary.image(
        'gt/landmark-regression',
        utils.tf_n_channel_rgb(gt_heatmap, n_channel),
        max_outputs=3)


def summary_predictions(data_eps, network_eps, is_training=True):
    predictions, _ = network_eps
    batch_size = tf.shape(predictions)[0]
    height = tf.shape(predictions)[1]
    width = tf.shape(predictions)[2]

    tf.summary.image(
        'predictions/batch',
        tf.map_fn(
            functools.partial(utils.tf_image_batch_to_grid, col_size=1),
            tf.transpose(tf.reshape(
                predictions, [batch_size, height, width, -1, 1]), [0, 3, 1, 2, 4])
        ),
        max_outputs=3)


def summary_output_image_batch(data_eps, network_eps, is_training=True):
    inputs, _ = network_eps

    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = tf.shape(inputs)[3]

    tf.summary.image(
        'predictions/pair',
        tf.map_fn(
            functools.partial(utils.tf_image_batch_to_grid, col_size=1),
            tf.transpose(tf.reshape(
                inputs, [batch_size, height, width, -1, 3]), [0, 3, 1, 2, 4])
        ),
        max_outputs=3)


class TBSummary(tf.keras.callbacks.TensorBoard):

    def set_model(self, model):

        # add learning rate summary
        optimizer = model.optimizer
        lr = optimizer.lr * \
            (1. / (1. + optimizer.decay * tf.to_float(optimizer.iterations)))
        tf.summary.scalar('learning_rate', lr)

        # add image summary
        for index, tf_inputs in enumerate(model.inputs):

            # normalise images
            tf_inputs_shape = tf_inputs.shape.as_list()
            if tf_inputs_shape[-1] not in [1, 3, 4]:
                tf_inputs = tf.transpose(tf_inputs, [0, 3, 1, 2])
                tf_inputs = tf.reshape(
                    tf_inputs, shape=[-1, tf_inputs_shape[1], tf_inputs_shape[2], 1])
                tf_inputs = utils.tf_image_batch_to_grid(tf_inputs)[None, ...]
            tf.summary.image('inputs_%02d' % index, tf_inputs)

        for index, tf_outputs in enumerate(model.outputs):
            tf_outputs_shape = tf_outputs.shape.as_list()
            if tf_outputs.shape.as_list()[-1] not in [1, 3, 4]:
                tf_outputs = tf.transpose(tf_outputs, [0, 3, 1, 2])
                tf_outputs = tf.reshape(
                    tf_outputs, shape=[-1, tf_outputs_shape[1], tf_outputs_shape[2], 1])
                tf_outputs = utils.tf_image_batch_to_grid(
                    tf_outputs)[None, ...]

            tf.summary.image('outputs_%02d' % index, tf_outputs)

        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        self.model.progress_bar.update(
            epoch,
            values=logs.items()
        )
        super().on_epoch_end(epoch, logs=logs)
