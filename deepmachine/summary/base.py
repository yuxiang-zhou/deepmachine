import tensorflow as tf
import functools
slim = tf.contrib.slim
from deepmachine import utils


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


def summary_iuv(data_eps, network_eps, is_training=True):

    iuv_gt = data_eps['iuv']
    _, states = network_eps
    iuv_pred = states['uv']

    iuv_gt_rgb = utils.tf_iuv_rgb(iuv_gt)
    iuv_pred_rgb = utils.tf_iuv_rgb(iuv_pred)

    # iuv summary
    tf.summary.image(
        'predictions/iuv',
        iuv_pred_rgb,
        max_outputs=3)

    tf.summary.image(
        'gt/iuv ',
        iuv_gt_rgb,
        max_outputs=3)


def summary_landmarks(data_eps, network_eps, is_training=True):
    gt_heatmap = data_eps['heatmap']
    predictions, _ = network_eps

    # landmarks summary
    tf.summary.image(
        'predictions/landmark-regression',
        tf.reduce_sum(predictions, -1)[..., None],
        max_outputs=3)

    tf.summary.image(
        'gt/landmark-regression',
        tf.reduce_sum(gt_heatmap, -1)[..., None],
        max_outputs=3)


def summary_predictions(data_eps, network_eps, is_training=True):
    predictions, _ = network_eps
    tf.summary.image(
        'predictions/batch',
        tf.map_fn(
            utils.tf_image_batch_to_grid,
            tf.transpose(predictions, [0, 3, 1, 2])[..., None]
        ),
        max_outputs=3)

def summary_cyclegan(data_eps, network_eps, is_training=True):
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