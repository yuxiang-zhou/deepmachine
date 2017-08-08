import tensorflow as tf
slim = tf.contrib.slim
from deepmachine import utils


def summary_input_image(data_eps, network_eps, is_training=True):
    tf.summary.image('images', data_eps['inputs'], max_outputs=4)


def summary_total_loss(data_eps, network_eps, is_training=True):
    if is_training:
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total_loss', total_loss)


def summary_input_image(data_eps, network_eps, is_training=True):
    uvs = data_eps['uv']
    tf.summary.image('uvs/h', uvs[..., :1])
    tf.summary.image('uvs/v', uvs[..., 1:])

def summary_iuv(data_eps, network_eps, is_training=True):

    iuv_gt = data_eps['iuv']
    _, states = network_eps
    iuv_pred = states[0]

    bsize = tf.shape(iuv_pred)[0]
    chsize = tf.shape(iuv_pred)[3]
    h = tf.shape(iuv_pred)[1]
    w = tf.shape(iuv_pred)[2]

    # iuv summary
    tf.summary.image(
        'predictions/cascade-regression',
        tf.map_fn(
            utils.tf_image_batch_to_grid,
            tf.transpose(iuv_pred, [0, 3, 1, 2])[..., None]
        ),
        max_outputs=4)

    tf.summary.image(
        'gt/cascade ',
        tf.map_fn(
            utils.tf_image_batch_to_grid,
            tf.transpose(iuv_gt, [0, 3, 1, 2])[..., None]
        ),
        max_outputs=4)


def summary_landmarks(data_eps, network_eps, is_training=True):
    gt_heatmap = data_eps['heatmap']
    predictions, _ = network_eps

    # landmarks summary
    tf.summary.image(
        'predictions/landmark-regression',
        tf.reduce_sum(predictions, -1)[..., None],
        max_outputs=4)

    tf.summary.image(
        'gt/landmark-regression ',
        tf.reduce_sum(gt_heatmap, -1)[..., None],
        max_outputs=4)
