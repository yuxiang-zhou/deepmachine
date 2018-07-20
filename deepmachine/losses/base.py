import tensorflow as tf

from . import helper
from .. import utils
from ..flags import FLAGS

def loss_heatmap_regression(y_true, y_pred, heatmap_weight=500):

    # landmark-regression losses
    weight_hm = utils.tf_get_weight(y_true, ng_w=0.1, ps_w=1.0) * heatmap_weight
    l2norm = tf.losses.mean_squared_error(
        y_pred, y_true, weights=weight_hm)

    tf.summary.scalar('losses/heatmap', l2norm)
    
    return l2norm


def loss_iuv_regression(y_true, y_pred, n_feature=26):
    n_feature = y_pred.shape.as_list()[-1] // 3

    # mask index cross-entropy loss
    uv_pred_idx = y_pred[..., :n_feature]  # 26 channels
    uv_gt_idx = y_true[..., :n_feature]  # 26 channels

    celoss = tf.losses.softmax_cross_entropy(uv_pred_idx, uv_gt_idx) * 5 

    # uv regression losses
    uv_pred = y_pred[..., n_feature:]  # 52 channels
    uv_gt = y_true[..., n_feature:]  # 52 channels

    weight_hm = uv_gt_idx * 100
    l1smooth_U = tf.losses.mean_squared_error(
        uv_pred[..., :n_feature], uv_gt[..., :n_feature], weights=weight_hm)
    l1smooth_V = tf.losses.mean_squared_error(
        uv_pred[..., n_feature:], uv_gt[..., n_feature:], weights=weight_hm)

    # losses summaries
    tf.summary.scalar('losses/u_y_pred', l1smooth_U)
    tf.summary.scalar('losses/v_y_pred', l1smooth_V)
    tf.summary.scalar('losses/uv_cross_entropy', celoss)

    return celoss + l1smooth_U + l1smooth_V