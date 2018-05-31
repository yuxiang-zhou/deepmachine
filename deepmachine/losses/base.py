import tensorflow as tf
slim = tf.contrib.slim

from . import helper
from .. import utils
from ..flags import FLAGS


def loss_landmark_regression(data_eps, network_eps, alpha=1.0, heatmap_weight=500, collection='regression_loss'):
    gt_heatmap = data_eps['heatmap']
    predictions, _ = network_eps

    # landmark-regression losses
    weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * heatmap_weight
    l2norm = slim.losses.mean_squared_error(
        predictions, gt_heatmap, weights=weight_hm * alpha)

    # losses summaries
    tf.summary.scalar('losses/lms_pred', l2norm)

    if collection is not None:
        tf.losses.add_loss(l2norm, loss_collection=collection)


def loss_landmark_reconstruction(data_eps, network_eps, alpha=1.0, heatmap_weight=500, collection='reconstruction_loss'):
    gt_heatmap = data_eps['heatmap']
    _, end_points = network_eps
    rec_heatmap = end_points[-1]

    # landmark-regression losses
    weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * heatmap_weight
    l2norm = slim.losses.mean_squared_error(
        rec_heatmap, gt_heatmap, weights=weight_hm * alpha)

    # losses summaries
    tf.summary.scalar('losses/lms_rec', l2norm)

    if collection is not None:
        tf.losses.add_loss(l2norm, loss_collection=collection)


def loss_stacked_landmark_regression(data_eps, network_eps, alpha=1.0, heatmap_weight=500):
    gt_heatmap = data_eps['heatmap']
    _, states = network_eps

    weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * heatmap_weight

    for idx_stack, predictions in enumerate(states):

        # landmark-regression losses
        l2norm = slim.losses.mean_squared_error(
            predictions, gt_heatmap, weights=weight_hm * alpha)

        # losses summaries
        tf.summary.scalar('losses/lms_stack_%02d' % idx_stack, l2norm)


def loss_iuv_regression(data_eps, network_eps, alpha=1.0, n_feature=26):
    cascade_gt = data_eps['iuv']
    _, states = network_eps

    cascade_prediction = states['uv']

    # mask index cross-entropy loss
    uv_pred_idx = cascade_prediction[..., :n_feature]  # 26 channels
    uv_gt_idx = cascade_gt[..., :n_feature]  # 26 channels

    celoss = slim.losses.softmax_cross_entropy(uv_pred_idx, uv_gt_idx) * 5  * alpha

    # uv regression losses
    uv_pred = cascade_prediction[..., n_feature:]  # 52 channels
    uv_gt = cascade_gt[..., n_feature:]  # 52 channels

    weight_hm = uv_gt_idx * 100
    l1smooth_U = helper.smooth_l1(
        uv_pred[..., :n_feature], uv_gt[..., :n_feature], weights=weight_hm * alpha)
    l1smooth_V = helper.smooth_l1(
        uv_pred[..., n_feature:], uv_gt[..., n_feature:], weights=weight_hm * alpha)

    # losses summaries
    tf.summary.scalar('losses/u_pred', l1smooth_U)
    tf.summary.scalar('losses/v_pred', l1smooth_V)
    tf.summary.scalar('losses/uv_cross_entropy', celoss)


def loss_uv_classification(data_eps, network_eps, alpha=1.0):
    k = FLAGS.quantization_step
    uvs = data_eps['uv']
    n_classes = k + 1
    _, states = network_eps
    logits = states['uv']

    for i, name in enumerate(['hor', 'ver']):
        gt = tf.to_int64(tf.floor(uvs[..., i] * k))
        gt = tf.reshape(gt, [-1])
        gt = slim.one_hot_encoding(gt, n_classes)
        class_loss = tf.contrib.losses.softmax_cross_entropy(
            tf.reshape(
                logits[..., i * n_classes: (i + 1) * n_classes],
                [-1, n_classes]
            ), gt)
        tf.summary.scalar(
            'losses/classification_loss_{}'.format(name), class_loss)