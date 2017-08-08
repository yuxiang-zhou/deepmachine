import tensorflow as tf
slim = tf.contrib.slim

from . import helper
from .. import utils
from ..flags import FLAGS

def loss_landmark_regression(data_eps, network_eps, heatmap_weight=500):
    gt_heatmap = data_eps['heatmap']
    predictions, _ = network_eps

    # landmark-regression losses
    weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * heatmap_weight
    l2norm = slim.losses.mean_squared_error(predictions, gt_heatmap, weights=weight_hm)

    # losses summaries
    tf.summary.scalar('losses/lms_pred', l2norm)


def loss_stacked_landmark_regression(data_eps, network_eps, heatmap_weight=500):
    gt_heatmap = data_eps['heatmap']
    _, states = network_eps

    weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * heatmap_weight

    for idx_stack, predictions in enumerate(states):

        # landmark-regression losses
        l2norm = slim.losses.mean_squared_error(predictions, gt_heatmap, weights=weight_hm)

        # losses summaries
        tf.summary.scalar('losses/lms_stack_%02d'%idx_stack, l2norm)


def loss_iuv_regression(data_eps, network_eps):
    cascade_gt = data_eps['iuv']
    _, states = network_eps

    cascade_prediction = states[0]

    # mask index cross-entropy loss
    uv_pred_idx = cascade_prediction[...,:26] # 26 channels
    uv_gt_idx = cascade_gt[...,:26] # 26 channels

    celoss = slim.losses.softmax_cross_entropy(uv_pred_idx, uv_gt_idx)


    # uv regression losses
    uv_pred = cascade_prediction[...,26:] # 52 channels
    uv_gt = cascade_gt[...,26:] # 52 channels

    weight_hm = uv_gt_idx * 100
    l1smooth_U = helper.smooth_l1(uv_pred[...,:26], uv_gt[...,:26], weights=weight_hm)
    l1smooth_V = helper.smooth_l1(uv_pred[...,26:], uv_gt[...,26:], weights=weight_hm)

    # losses summaries
    tf.summary.scalar('losses/u_pred', l1smooth_U)
    tf.summary.scalar('losses/v_pred', l1smooth_V)
    tf.summary.scalar('losses/uv_cross_entropy', celoss)

def loss_uv_classification(data_eps, network_eps):
    k = FLAGS.quantization_step
    uvs = data_eps['uv']
    n_classes = k+1
    _, states = network_eps
    logits, *_ = states

    for i, name in enumerate(['hor', 'ver']):
        gt = tf.to_int64(tf.floor(uvs[..., i] * k))
        gt = tf.reshape(gt, [-1])
        gt = slim.one_hot_encoding(gt, n_classes)
        class_loss = tf.contrib.losses.softmax_cross_entropy(
            tf.reshape(logits[..., i * n_classes: (i+1) * n_classes], [-1, n_classes]), gt)
        tf.summary.scalar('losses/classification loss [{}]'.format(name), class_loss)
